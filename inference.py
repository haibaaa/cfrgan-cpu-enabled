import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import glob
from tqdm import tqdm
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

def normalize(img):
    return (img-0.5)*2

def check_face_model(face_model_path):
    if not os.path.exists(face_model_path):
        print(f"Error: Face model file not found at {face_model_path}")
        print("\nPlease download the BFM_model_80.mat file and place it in the mmRegressor/BFM/ directory.")
        print("You can download it from: https://faces.dmi.unibas.ch/bfm/bfm2017.html")
        print("After downloading, place the file at: mmRegressor/BFM/BFM_model_80.mat")
        return False
    return True

def inference(img_path, save_path, aligner, generator_path, estimator_path, face_model_path, batch_size=4):
    # Check if face model exists
    if not check_face_model(face_model_path):
        return

    device = torch.device('cpu')
    
    # Initialize models
    netG = CFRNet().to(device)
    estimator3d = Estimator3D(batch_size=batch_size, render_size=224, test=True, estimator_path=estimator_path, face_model_path=face_model_path)
    
    # Load generator weights
    trained_weights = torch.load(generator_path, map_location=device)
    own_state = netG.state_dict()
    
    # Copy weights from trained model to current model
    for name, param in trained_weights.items():
        if name.startswith('module.'):
            name = name[7:]  # Remove 'module.' prefix
        if name in own_state:
            own_state[name].copy_(param)
    
    netG.eval()
    
    # Process images
    img_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
    for k in tqdm(range(0, len(img_list), batch_size)):
        until = k + batch_size
        if until > len(img_list):
            until = len(img_list)
            
        # Load and process images
        input_img = estimator3d.load_images(img_list[k:until])
        if input_img is None:
            continue
            
        # Generate testing pairs
        rotated, guidance = estimator3d.generate_testing_pairs(input_img, pose=[5.0, 0.0, 0.0])
        
        # Normalize and permute dimensions
        rotated = normalize(rotated[...,[2,1,0]].permute(0,3,1,2).contiguous())
        guidance = normalize(guidance[...,[2,1,0]].permute(0,3,1,2).contiguous())
        
        # Move to device
        rotated = rotated.to(device)
        guidance = guidance.to(device)
        
        # Run inference
        with torch.no_grad():
            output, occ_mask = netG(rotated, guidance)
            
        # Process output
        output = (output / 2) + 0.5
        output = (output.permute(0,2,3,1)*255).cpu().detach().numpy().astype('uint8')
        
        # Save results
        for i in range(rotated.shape[0]):
            save_file = os.path.join(save_path, os.path.basename(img_list[k+i]))
            cv2.imwrite(save_file, cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

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