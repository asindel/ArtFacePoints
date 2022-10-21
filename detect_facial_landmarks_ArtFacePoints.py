# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 2021

@author: Aline Sindel
"""

import argparse
import os
import torch
import cv2

from test_dataset import ImageDataset
from network import ArtFacePoints
from data_utils import visualizeLandmarks, write_landmarks_as_pts_format, set_random_seeds, getRelPaths

        
def main(opt):

    set_random_seeds(0)
    
    opt.use_cuda = (not opt.no_cuda and torch.cuda.is_available())
    device = torch.device("cuda:0" if opt.use_cuda else "cpu")   
    
    test_dataset = ImageDataset(opt.dataset_dir, opt.patch_size)    
    len_test_set = len(test_dataset)    
    print("Test set contains {} images".format(len_test_set))    
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=opt.batch_size, shuffle=False, num_workers=0)

    facial_net = ArtFacePoints(opt, device)
    facial_net.load_weights(opt.model_dir, opt.model_epoch, opt.load_checkpoint)
    facial_net.eval()
  
    img_paths = test_dataset.image_filenames
    img_files = getRelPaths(img_paths, opt.data_dir)  

    print("Run ArtFacePoints and save predictions and visualizations to ", opt.out_dir)
    
    for i, images in enumerate(test_loader):        
          images = images.to(device)         
          
          with torch.no_grad():
              predictions, global_predictions, LR_images = facial_net(images)              
              
              # save predictions and landmark visualizations to disk
              B = images.size(0)
              for b in range(B):
                  LR_image = LR_images[b].unsqueeze(dim=0)
                  image = images[b].unsqueeze(dim=0)
                  global_prediction = global_predictions[b].unsqueeze(dim=0)
                  prediction = predictions[b].unsqueeze(dim=0)
                  
                  if opt.save_global_pred:
                      #visualize global facial landmark results
                      global_img_landmark_pred = visualizeLandmarks(LR_image, global_prediction, in_size=opt.region_size, line_thickness=1, point_radius=3)
                      global_img_landmark_pred = cv2.cvtColor(global_img_landmark_pred,cv2.COLOR_RGB2BGR)              
                      global_out_patchfile = os.path.join(opt.out_dir, "overlay_pred_global_landmarks", img_files[i*B+b])
                      out_subdirs,_ = os.path.split(global_out_patchfile)
                      if os.path.exists(out_subdirs)==False:
                          os.makedirs(out_subdirs)
                      cv2.imwrite(global_out_patchfile, global_img_landmark_pred)
                  
                  #visualize region facial network refinement results               
                  img_landmark_pred = visualizeLandmarks(image, prediction, in_size=opt.patch_size, line_thickness=5, point_radius=10)
                  img_landmark_pred = cv2.cvtColor(img_landmark_pred,cv2.COLOR_RGB2BGR)              
                  out_patchfile = os.path.join(opt.out_dir, "overlay_pred_landmarks", img_files[i*B+b])
                  out_subdirs,_ = os.path.split(out_patchfile)
                  if os.path.exists(out_subdirs)==False:
                      os.makedirs(out_subdirs)
                  cv2.imwrite(out_patchfile, img_landmark_pred)  
                  
                  if opt.save_global_pred:
                      #save global landmarks predictions (in .pts format)
                      np_global_prediction = (global_prediction.detach().cpu().numpy().squeeze(0) + 0.5)*opt.patch_size*opt.scale
                      #print(np_global_prediction)
                      out_global_pred_ptsfile = os.path.join(opt.out_dir, "pred_global_landmarks", os.path.splitext(img_files[i*B+b])[0]+".pts")
                      out_subdirs,_ = os.path.split(out_global_pred_ptsfile)
                      if os.path.exists(out_subdirs)==False:
                          os.makedirs(out_subdirs)
                      write_landmarks_as_pts_format(np_global_prediction, out_global_pred_ptsfile)
                  
                  #save ArtFacePoints landmarks predictions (in .pts format (as in menpo lib), note: 1 pixel offset will be added, since .pts format is using Matlab indexing)
                  np_prediction = (prediction.detach().cpu().numpy().squeeze(0) + 0.5)*opt.patch_size*opt.scale
                  out_pred_ptsfile = os.path.join(opt.out_dir, "pred_landmarks", os.path.splitext(img_files[i*B+b])[0]+".pts")
                  out_subdirs,_ = os.path.split(out_pred_ptsfile)
                  if os.path.exists(out_subdirs)==False:
                      os.makedirs(out_subdirs)                 
                  write_landmarks_as_pts_format(np_prediction, out_pred_ptsfile)
    
    print('Total number of test images: {}'.format(len(test_dataset)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('facial landmark detection') 
    parser.add_argument('--root_dir', help='path to root',type=str, default='D:/Project/Code/FaceDetection/ArtFacePoints') #put your path here
    parser.add_argument('--data_dir', help='path to dataset root',type=str, default='D:/Project/Data/FaceDatabases/StyleArtFacesDataset/test') #put your path here
    parser.add_argument('--dataset_dir', help='path to dataset dir (can be subfolder)',type=str, default='D:/Project/Data/FaceDatabases/StyleArtFacesDataset/test/Paintings') #put your path here
    parser.add_argument('--model_dir', help='trained region model dir',type=str, default='D:/Project/Code/FaceDetection/ArtFacePoints/weights') #put your path here
    parser.add_argument('--model_epoch', type=str, default='best', help='epoch of trained region model')
    parser.add_argument('--load_checkpoint', type=bool, default=True, help='model_path is path to checkpoint (True) or path to state dict (False)')
    parser.add_argument('--out_dir', help='path to dataset dir',type=str, default='E:/FacialLandmarks/ArtDataset/test/ArtFacePoints') #put your path here
    parser.add_argument('--save_global_pred', action='store_true', default=False, help="Additionally save the global predictions of ArtFacePoints (first stage)")
    parser.add_argument('--scale', type=float, default=1, help='scaling factor for saving predictions e.g. 1 or 0.25')
    parser.add_argument('--patch_size', type=int, default=1024, help='input patch size')
    parser.add_argument('--global_patch_size', type=int, default=256, help='global network input patch size')
    parser.add_argument('--region_size', type=int, default=256, help='region network input patch size')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--which_network', type=str, default='resnet_9blocks', help='unet or resnet_6blocks or resnet_9blocks')
    parser.add_argument('--up_type', type=str, default= 'bicubic', help='transposed conv or bicubic + conv') 
    parser.add_argument('--input_nc', type=int, default=3, help='network input channel')
    parser.add_argument('--output_nc', type=int, default=68, help='network output channel') 
    parser.add_argument('--upsampling_mode', type=str, default='bicubic', help='upsampling method for landmark refinement: bicubic, linear')
    parser.add_argument('--norm', type=str, default= 'batch', help='normalization layer: batch') 
    parser.add_argument('--init_type', type=str, default= 'normal', help='initialization of network: normal') 
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training (please use CUDA_VISIBLE_DEVICES to select GPU)')
    args = parser.parse_args()     
    
    main(args)    

