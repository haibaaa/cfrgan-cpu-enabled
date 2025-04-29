#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])

from mmRegressor.network.resnet50_task import resnet50_use
from mmRegressor.load_data import BFM
from mmRegressor.reconstruct_mesh import Reconstruction, Compute_rotation_matrix, Projection_layer, _need_const
from faceParsing.model import BiSeNet
from tools.ops import erosion, dilation, blur, SCDiffer

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import cv2
import glob
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    HardPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    Textures
)
from tqdm import tqdm
import math

class Estimator3D(object):
    def __init__(self, batch_size=1, render_size=224, test=True, estimator_path=None, face_model_path="mmRegressor/BFM/BFM_model_80.mat", back_white=False):
        self.is_cuda = False
        self.device = torch.device('cpu')
        self.render_size = render_size
        self.batch_size = batch_size
        self.estimator_path = estimator_path
        self.face_model_path = face_model_path
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.load_3dmm_models(test)
        tri = self.face_model.tri
        tri = np.expand_dims(tri, 0)
        self.tri = torch.FloatTensor(tri).repeat(batch_size, 1, 1).to(self.device)
        self.skin_mask = -1 * self.face_model.skin_mask.unsqueeze(-1).to(self.device)
        self.init_renderer(back_white)

    def load_3dmm_models(self, test=True):
        if not os.path.exists(self.face_model_path):
            raise FileNotFoundError(f"Face model .mat file not found: {self.face_model_path}")
        self.face_model = BFM(self.face_model_path)

        regressor = resnet50_use()
        if self.estimator_path is None:
            raise ValueError("Estimator .pth file path must be provided")
        regressor.load_state_dict(torch.load(self.estimator_path, map_location=self.device))
        if test:
            regressor.eval()
            for param in regressor.parameters():
                param.requires_grad = False
        self.regressor = regressor.to(self.device)

    def init_renderer(self, back_white):
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        if back_white:
            blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        R, T = look_at_view_transform(eye=[[0, 0, 10]], at=[[0, 0, 0]], up=[[0, 1, 0]], device=self.device)
        self.R, self.T = R, T
        camera = FoVPerspectiveCameras(znear=0.01, zfar=50.0, aspect_ratio=1.0, fov=12.5936, R=R, T=T, device=self.device)
        lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]], device=self.device, location=[[0.0, 0.0, 1e-5]])
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=self.render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=True
                )
            ),
            shader=HardPhongShader(cameras=camera, device=self.device, lights=lights, blend_params=blend_params)
        )

    def regress_3dmm(self, img):
        arr_coef = self.regressor(img)
        coef = torch.cat(arr_coef, 1)
        return coef

    def reconstruct(self, coef, test=True):
        face_shape, _, face_color, _, face_projection, z_buffer, gamma, front_face = Reconstruction(coef, self.face_model)
        verts_rgb = face_color[..., [2, 1, 0]]
        mesh = Meshes(verts=face_shape, faces=self.tri[:face_shape.shape[0], ...], textures=Textures(verts_rgb=verts_rgb))
        rendered = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T)
        rendered = torch.clamp(rendered, 0.0, 1.0)
        landmarks_2d = torch.zeros_like(face_projection).to(self.device)
        landmarks_2d[..., 0] = torch.clamp(face_projection[..., 0].clone(), 0, self.render_size - 1)
        landmarks_2d[..., 1] = torch.clamp(face_projection[..., 1].clone(), 0, self.render_size - 1)
        landmarks_2d[..., 1] = self.render_size - landmarks_2d[..., 1].clone() - 1
        landmarks_2d = landmarks_2d[:, self.face_model.keypoints, :]
        if test:
            return rendered, landmarks_2d, face_shape, face_color, front_face, z_buffer
        tex_mean = torch.sum(face_color * self.skin_mask) / torch.sum(self.skin_mask)
        ref_loss = torch.sum(torch.square((face_color - tex_mean) * self.skin_mask)) / (face_color.shape[0] * torch.sum(self.skin_mask))
        gamma = gamma.view(-1, 3, 9)
        gamma_mean = torch.mean(gamma, dim=1, keepdim=True)
        gamma_loss = torch.mean(torch.square(gamma - gamma_mean))
        return rendered, landmarks_2d, ref_loss, gamma_loss

    def estimate_and_reconstruct(self, img):
        coef = self.regress_3dmm(img)
        return self.reconstruct(coef, test=True)

    def get_colors_from_image(self, image, proj, z_buffer, scaling=True, normalized=False, reverse=True, z_cut=None):
        # Ensure image is in the correct format (batch, height, width, channels)
        if len(image.shape) == 4 and image.shape[1] == 3:  # If in (batch, channels, height, width) format
            image = image.permute(0, 2, 3, 1)  # Convert to (batch, height, width, channels)
        elif len(image.shape) == 3:  # If single image
            image = image.unsqueeze(0)  # Add batch dimension
            
        batch_size, h, w, channels = image.shape
        
        # Ensure proj is a PyTorch tensor
        if isinstance(proj, np.ndarray):
            proj = torch.from_numpy(proj).float().to(self.device)
        elif not isinstance(proj, torch.Tensor):
            proj = torch.tensor(proj, dtype=torch.float32, device=self.device)
        else:
            proj = proj.float().to(self.device)
            
        if scaling:
            proj *= self.render_size / 224
            
        proj[..., 0] = torch.clamp(proj[..., 0], 0, w - 1)
        proj[..., 1] = torch.clamp(proj[..., 1], 0, h - 1)
        
        if reverse:
            proj[..., 1] = h - proj[..., 1] - 1
            
        idx = torch.round(proj).type(torch.long)
        colors = []
        
        for k in range(batch_size):
            # Get the current image and indices
            curr_img = image[k]  # Shape: (h, w, channels)
            curr_idx = idx[k]    # Shape: (num_points, 2)
            
            # Ensure curr_idx has the correct shape
            if len(curr_idx.shape) == 1:
                # If it's a 1D tensor with 3 elements, it's likely (x, y, z)
                if curr_idx.shape[0] == 3:
                    curr_idx = curr_idx[:2].unsqueeze(0)  # Take only x,y and add batch dimension
                else:
                    raise ValueError(f"Unexpected index tensor shape: {curr_idx.shape}")
            
            # Extract colors using advanced indexing
            y_indices = curr_idx[:, 1].clamp(0, h-1)
            x_indices = curr_idx[:, 0].clamp(0, w-1)
            curr_colors = curr_img[y_indices, x_indices]  # Shape: (num_points, channels)
            colors.append(curr_colors.unsqueeze(0))
            
        colors = torch.cat(colors, dim=0)  # Shape: (batch_size, num_points, channels)
        
        z_buffer = z_buffer.squeeze(2)
        if z_cut is not None:
            colors = colors.repeat(2, 1, 1)
            colors[:z_buffer.shape[0]][z_buffer < z_cut] = 0.0
            
        if not normalized:
            colors = colors / 255
            
        return colors.to(self.device)

    def get_occlusion_mask(self, model, scd, rot, gui, ori_img, obj=False):
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        parsed = model(F.interpolate((ori_img - mean) / std, (512, 512), mode='bilinear', align_corners=True))
        parsed = torch.argmax(F.interpolate(parsed, (self.render_size, self.render_size), mode='bilinear', align_corners=True), dim=1, keepdim=True)

        guidance_gray = torch.mean(gui, dim=1, keepdim=True)
        guidance_noise = (guidance_gray < 0.04)

        coarse_occ_mask = torch.ones_like(guidance_noise, dtype=rot.dtype).to(self.device)
        idx = (parsed == 1) | (parsed == 2) | ((parsed >= 4) & (parsed <= 7)) | ((parsed >= 10) & (parsed <= 12))
        coarse_occ_mask[idx] = 0.0

        eye_idx = (parsed == 4) | (parsed == 5)

        parsed_g = model(F.interpolate((gui - mean) / std, (512, 512), mode='bilinear', align_corners=True))
        parsed_g = torch.argmax(F.interpolate(parsed_g, (self.render_size, self.render_size), mode='bilinear', align_corners=True), dim=1, keepdim=True)
        parsed_g = (parsed_g == 4) | (parsed_g == 5)

        closed_eye_batch = torch.ge(torch.sum(torch.logical_and(parsed_g, parsed == 1).float(), dim=[1, 2, 3]) / (torch.sum(parsed_g.float(), dim=[1, 2, 3]) + 1e-7), 0.5)
        closed_eye_batch = closed_eye_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(eye_idx)
        eye_idx = torch.logical_or(eye_idx, torch.logical_and(parsed_g, closed_eye_batch))

        eye_idx = dilation(eye_idx.float(), filter_size=5).bool()
        coarse_occ_mask[eye_idx] = 0.0
        coarse_occ_mask[guidance_noise] = 0.0

        background = torch.zeros_like(guidance_noise, dtype=gui.dtype).to(self.device)
        background[guidance_noise] = 1.0
        background[eye_idx] = 0.0
        background[torch.logical_or(parsed == 3, parsed == 10)] = 0.0

        gui_with_bg = ori_img * background + gui * (1. - background)

        background = erosion(background, filter_size=2)
        background = blur(background, filter_size=7)
        edge = torch.zeros_like(background).to(self.device)
        edge[torch.logical_and(background > 0.0, background < 0.8)] = 1.0

        eye_mask = torch.zeros_like(eye_idx, dtype=rot.dtype).to(self.device)
        eye_mask[eye_idx] = 1.0

        if not obj:
            diff = scd(rot, gui, alpha=0.33) * (1. - eye_mask)
            eye_diff = scd(rot, ori_img, alpha=0.5) * eye_mask
            adaptive_guide = diff + (coarse_occ_mask + eye_diff)
            adaptive_guide = torch.clamp(adaptive_guide, 0.0, 1.0)
            adaptive_guide *= (1. - edge)
            adaptive_guide = torch.round(adaptive_guide)
            return adaptive_guide
        else:
            diff = torch.round(scd(ori_img, gui_with_bg)) * (1. - eye_mask)
            brow_lip_mask = torch.ones_like(eye_idx, dtype=rot.dtype).to(self.device)
            brow_lip_mask[torch.logical_or(torch.logical_or(parsed == 6, parsed == 7), torch.logical_or(parsed == 11, parsed == 12))] = 0.6
            diff[parsed == 10] = 0.0
            diff *= brow_lip_mask
            guide_obj = diff + coarse_occ_mask
            guide_obj = torch.clamp(guide_obj, 0.0, 1.0)
            face_mask = torch.sum(guidance_gray >= 0.04, dim=[1, 2, 3])
            occ_cands = torch.sum(torch.round(guide_obj + 0.1), dim=[1, 2, 3])
            occ_rates = occ_cands / face_mask
            for b in range(occ_rates.shape[0]):
                if occ_rates[b] < 0.09:
                    guide_obj[b] *= 0.1
            return guide_obj

    def swap_and_rotate_and_render(self, image, parsing_net, scd):
        coef = self.regress_3dmm(image)
        face_shape, ori_angles, _, face_color, face_projection, z_buffer, _, front_face = Reconstruction(coef, self.face_model)
        
        # Ensure face_shape and face_color have the same number of vertices
        num_vertices = face_shape.shape[1]
        color_from_img = self.get_colors_from_image(image, face_projection, z_buffer, normalized=True)
        
        # Ensure color_from_img has the correct number of vertices
        if color_from_img.shape[1] != num_vertices:
            # If we have fewer colors than vertices, pad with zeros
            if color_from_img.shape[1] < num_vertices:
                padding = torch.zeros((color_from_img.shape[0], num_vertices - color_from_img.shape[1], color_from_img.shape[2]), 
                                    device=color_from_img.device)
                color_from_img = torch.cat([color_from_img, padding], dim=1)
            # If we have more colors than vertices, truncate
            else:
                color_from_img = color_from_img[:, :num_vertices, :]
        
        # Create mesh with the colors
        rot = Meshes(verts=face_shape, faces=self.tri[:image.shape[0], ...], 
                    textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot, 0.0, 1.0)
        
        # Create guidance mesh
        gui = Meshes(verts=face_shape, faces=self.tri[:image.shape[0], ...], 
                    textures=Textures(verts_rgb=face_color[..., [2, 1, 0]]))
        gui = self.phong_renderer(meshes_world=gui, R=self.R, T=self.T)
        gui = torch.clamp(gui, 0.0, 1.0)
        
        rot_ori = rot.permute(0, 3, 1, 2)[:, [2, 1, 0], ...]
        gui_ori = gui.permute(0, 3, 1, 2)[:, [2, 1, 0], ...]
        
        mask_only_obj = self.get_occlusion_mask(parsing_net, scd, rot, gui_ori, image[:, [2, 1, 0], ...], obj=True)
        blur_mask_o = blur(dilation(mask_only_obj, filter_size=3), filter_size=3)
        
        rot = image[:, [2, 1, 0], ...] * (1. - blur_mask_o) + gui_ori * blur_mask_o
        gui = gui_ori * (1. - mask_only_obj) + rot_ori * mask_only_obj
        gui = gui * (1. - blur_mask_o) + blur_mask_o * blur(gui, filter_size=3)
        
        # Get colors for rotated mesh
        color_from_img = self.get_colors_from_image(rot, face_projection, z_buffer, normalized=True, reverse=False, scaling=False)
        
        # Ensure color_from_img has the correct number of vertices
        if color_from_img.shape[1] != num_vertices:
            if color_from_img.shape[1] < num_vertices:
                padding = torch.zeros((color_from_img.shape[0], num_vertices - color_from_img.shape[1], color_from_img.shape[2]), 
                                    device=color_from_img.device)
                color_from_img = torch.cat([color_from_img, padding], dim=1)
            else:
                color_from_img = color_from_img[:, :num_vertices, :]
        
        angles = torch.rand(rot.shape[0], 3).to(self.device)
        angles[:, 1] = (-math.pi * 90 / 180 - math.pi * 90 / 180) * angles[:, 1] + math.pi * 90 / 180
        angles[:, [0, 2]] = (-math.pi / 12 - math.pi / 12) * angles[:, [0, 2]] + math.pi / 12
        angles[:, 1] = torch.clamp(ori_angles[:, 1] + angles[:, 1], -math.pi / 2, math.pi / 2)
        angles[:, [0, 2]] = torch.clamp(ori_angles[:, [0, 2]] + angles[:, [0, 2]], -math.pi / 6, math.pi / 6)
        
        rotation_m = Compute_rotation_matrix(angles)
        rotated_shape = torch.matmul(front_face, rotation_m.to(self.device))
        
        # Create rotated mesh with the colors
        rot = Meshes(verts=rotated_shape, faces=self.tri[:rot.shape[0], ...], 
                    textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot[..., :3], 0.0, 1.0)
        
        if _need_const.gpu_p_matrix is None:
            _need_const.gpu_p_matrix = _need_const.p_matrix.to(self.device)
        p_matrix = _need_const.gpu_p_matrix.expand(rotated_shape.shape[0], 3, 3)
        
        aug_projection = rotated_shape.clone().detach()
        aug_projection[:, :, 2] = _need_const.cam_pos - aug_projection[:, :, 2]
        aug_projection = aug_projection.bmm(p_matrix.permute(0, 2, 1))
        face_projection = aug_projection[:, :, 0:2] / aug_projection[:, :, 2:]
        z_buffer = _need_const.cam_pos - aug_projection[:, :, 2:]
        
        # Get final colors
        color_from_img = self.get_colors_from_image(rot, face_projection, z_buffer, normalized=True)
        
        # Ensure color_from_img has the correct number of vertices
        if color_from_img.shape[1] != num_vertices:
            if color_from_img.shape[1] < num_vertices:
                padding = torch.zeros((color_from_img.shape[0], num_vertices - color_from_img.shape[1], color_from_img.shape[2]), 
                                    device=color_from_img.device)
                color_from_img = torch.cat([color_from_img, padding], dim=1)
            else:
                color_from_img = color_from_img[:, :num_vertices, :]
        
        # Create final mesh
        rot = Meshes(verts=face_shape, faces=self.tri[:rot.shape[0], ...], 
                    textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot[..., :3], 0.0, 1.0)
        
        occ_mask = self.get_occlusion_mask(parsing_net, scd, rot.permute(0, 3, 1, 2), gui, image[:, [2, 1, 0], ...], obj=False)
        occ_mask = (occ_mask * 3 + mask_only_obj) / 4
        
        return rot, gui.permute(0, 2, 3, 1), occ_mask

    def generate_testing_pairs(self, image, pose=[5.0, 0.0, 0.0], front=False, landmark=False):
        coef = self.regress_3dmm(image)
        face_shape, angles, _, face_color, _, face_projection, z_buffer, front_face = Reconstruction(coef, self.face_model)
        color_from_img = self.get_colors_from_image(image, face_projection, z_buffer, normalized=True)
        if not front:
            pose = torch.FloatTensor(pose).to(self.device)
            angles = torch.zeros(coef.shape[0], 3).to(self.device)
            angles[:] = math.pi * pose / 180
            rotation_m = Compute_rotation_matrix(angles)
        else:
            angles[:, [1, 2]] = 0.0
            rotation_m = Compute_rotation_matrix(angles)
        rotated_shape = torch.matmul(front_face, rotation_m.to(self.device))
        rotated = Meshes(verts=rotated_shape, faces=self.tri[:image.shape[0], ...], textures=Textures(verts_rgb=color_from_img))
        rotated = self.phong_renderer(meshes_world=rotated, R=self.R, T=self.T)
        rotated = torch.clamp(rotated, 0.0, 1.0)[..., :3]
        guidance = Meshes(verts=rotated_shape, faces=self.tri[:image.shape[0], ...], textures=Textures(verts_rgb=face_color[..., [2, 1, 0]]))
        guidance = self.phong_renderer(meshes_world=guidance, R=self.R, T=self.T)
        guidance = torch.clamp(guidance, 0.0, 1.0)[..., :3]
        if landmark:
            face_projection, _ = Projection_layer(front_face, rotation_m, rotation_m.new_zeros(angles.shape[0], 3))
            landmarks_2d = torch.zeros_like(face_projection).to(self.device)
            landmarks_2d[..., 0] = torch.clamp(face_projection[..., 0].clone(), 0, self.render_size - 1)
            landmarks_2d[..., 1] = torch.clamp(face_projection[..., 1].clone(), 0, self.render_size - 1)
            landmarks_2d[..., 1] = self.render_size - landmarks_2d[..., 1].clone() - 1
            landmarks_2d = landmarks_2d[:, self.face_model.keypoints, :]
            return rotated, guidance, landmarks_2d
        return rotated, guidance

    def load_images(self, img_list):
        input_img = []
        for filename in img_list:
            img = cv2.imread(filename)
            if img is None:
                print(f"Warning: Failed to load image {filename}")
                continue
            if img.shape[0] != self.render_size:
                img = cv2.resize(img, (self.render_size, self.render_size), cv2.INTER_AREA)
            if img.shape[2] == 4:
                img = img[..., :3]
            img = self.to_tensor(img)
            input_img.append(img.unsqueeze(0))
        if not input_img:
            print("Warning: No valid images loaded")
            return None
        input_img = torch.cat(input_img).to(self.device)
        return input_img

if __name__ == '__main__':
    batch_size = 2
    estimator_path = "saved_models/trained_weights_occ_3d.pth"
    face_model_path = "mmRegressor/BFM/BFM_model_80.mat"
    parsing_model_path = "faceParsing/model_final_diss.pth"
    input_img_path = "./test_imgs/input"
    save_path = "./test_imgs/output"

    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(parsing_model_path, map_location='cpu'))
    bisenet.eval()
    bisenet = bisenet.to('cpu')
    estimator = Estimator3D(batch_size=batch_size, render_size=224, test=True, estimator_path=estimator_path, face_model_path=face_model_path)
    scd = SCDiffer()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_list = glob.glob(os.path.join(input_img_path, '*.jpg'))
    print('The number of images:', len(img_list))
    for i in tqdm(range(0, len(img_list), batch_size)):
        until = i + batch_size
        if until > len(img_list):
            until = len(img_list)
        input_img = estimator.load_images(img_list[i:until])
        if input_img is None:
            continue
        rot, gui, occ_mask = estimator.swap_and_rotate_and_render(input_img, bisenet, scd)
        occ_mask = occ_mask.permute(0, 2, 3, 1).cpu().numpy() * 255.0
        gui = gui.cpu().numpy() * 255.0
        rot = rot.cpu().numpy() * 255.0
        for k in range(rot.shape[0]):
            cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[i + k])[:-4] + '_occ.jpg'), occ_mask[k])
            cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[i + k])[:-4] + '_rot.jpg'), cv2.cvtColor(rot[k], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[i + k])[:-4] + '_gui.jpg'), cv2.cvtColor(gui[k], cv2.COLOR_RGB2BGR))