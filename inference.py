import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import glob
from generate_pairs import Estimator3D
from model.layers import SNConv, ResnetBlock, StridedGatedConv, ConvTransINAct, MixingLayer, AFD

class CFRNet(torch.nn.Module):
    def __init__(self):
        super(CFRNet, self).__init__()
        self.conv_r = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            SNConv(3, 64, kernel_size=7, padding=0),
            torch.nn.InstanceNorm2d(64), torch.nn.LeakyReLU(0.2, True),
        )
        self.conv_g = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            SNConv(3, 64, kernel_size=7, padding=0),
            torch.nn.InstanceNorm2d(64), torch.nn.LeakyReLU(0.2, True),
        )

        self.afd = AFD(64, 64)
        
        self.downsample = torch.nn.ModuleList([
            StridedGatedConv(64, 128, kernel_size=3), # 128
            StridedGatedConv(128, 256, kernel_size=3), # 64
        ])

        blocks = []
        for _ in range(9):
            blocks.append(ResnetBlock(256, gate=True))
        self.resblocks = torch.nn.Sequential(*blocks)

        self.upsample = torch.nn.Sequential(
            ConvTransINAct(256, 128, ks=4, stride=2, padding=1),
            ConvTransINAct(128, 64, ks=4, stride=2, padding=1),
        )

        self.last_conv = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            SNConv(64, 3, kernel_size=7, padding=0),
            torch.nn.Tanh()
        )

        self.mixing = MixingLayer(dims=[64, 128, 256])

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, rotated, guidance, wo_mask=False):
        # Normalize
        rotated = self.conv_r(rotated)
        guidance = self.conv_g(guidance)

        diff = self.afd(rotated, guidance)
        attn = self.sigmoid(diff)

        out, gate1 = self.downsample[0](rotated*(1.-attn)) # 128
        out, gate2 = self.downsample[1](out) # 64

        out = self.resblocks(out)
        out = self.upsample(out)
        out = self.last_conv(out)
        
        if wo_mask:
            return out

        mask = self.mixing(diff, gate1, gate2)
        
        return out, mask

def inference(img_path, save_path, aligner, generator_path, estimator_path, face_model_path, batch_size=4):
    device = torch.device('cpu')
    netG = CFRNet()
    
    # Load state dict and handle module prefix
    state_dict = torch.load(generator_path, map_location=device)
    # Remove 'module.' prefix from keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    netG.load_state_dict(new_state_dict)
    
    netG.eval()
    netG = netG.to(device)
    
    estimator3d = Estimator3D(batch_size=batch_size, render_size=224, test=True, estimator_path=estimator_path, face_model_path=face_model_path)
    
    img_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
    for i in range(0, len(img_list), batch_size):
        batch_imgs = img_list[i:i+batch_size]
        input_img = estimator3d.load_images(batch_imgs)
        if input_img is None:
            continue
        with torch.no_grad():
            output = netG(input_img, input_img, wo_mask=True)  # Using same image for both rotated and guidance
        for j, img_file in enumerate(batch_imgs):
            save_file = os.path.join(save_path, os.path.basename(img_file))
            save_img = output[j].permute(1, 2, 0).cpu().numpy()
            save_img = (save_img * 255).astype(np.uint8)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_file, save_img)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--aligner', type=str, default='None')
    parser.add_argument('--generator_path', type=str, default="saved_models/CFRNet_G_ep55_vgg.pth", help="Generator model path")
    parser.add_argument('--estimator_path', type=str, default="saved_models/trained_weights_occ_3d.pth", help="3D estimator model path")
    parser.add_argument('--face_model_path', type=str, default="mmRegressor/BFM/BFM_model_80.mat")
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    inference(args.img_path, args.save_path, args.aligner, args.generator_path, args.estimator_path, args.face_model_path, args.batch_size)