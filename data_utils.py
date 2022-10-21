# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:49:20 2021

@author: Aline Sindel
"""
import os
import cv2
import numpy as np
import torch 
import torch.nn.functional as F
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF"])

def getRelPaths(img_files, root):
    rel_files = []
    for file in img_files:
        rel_files.append(os.path.relpath(file, root))
    return rel_files

def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    random.seed(random_seed)      
    np.random.seed(random_seed)

def tensor2imRGB(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    #based on tensor2im from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
    
def visualizeLandmarks(tensor_images, tensor_landmarks, in_size=1024, line_thickness=3, point_radius=5):
    landmarks_np = tensor_landmarks.clone()
    landmarks_np = landmarks_np[0]
    landmarks_np = landmarks_np.detach().cpu().numpy()  
    landmarks_np =  (landmarks_np + 0.5) * in_size
      
    img = tensor2imRGB(tensor_images)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = drawAnnotations(img, landmarks_np.astype(np.int32), None, False, line_thickness, point_radius)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img  

def defineRegionBatch(landmarks, offset_ratio, with_one_pixel_shift, device):
    B = landmarks.size(0)
    pts1, ind1 = landmarks.min(dim=1)
    pts2, ind2 = landmarks.max(dim=1)
    
    left = torch.round(pts1[:,0]) 
    top = torch.round(pts1[:,1]) 
    right = torch.round(pts2[:,0])
    bottom = torch.round(pts2[:,1])
    region_width = torch.maximum(right - left, bottom - top)
    #add padding
    region_width += (offset_ratio*region_width).int()
    
    #bbox: get center of bbox
    c_x = left + (0.5*(right-left)).int()
    c_y = top + (0.5*(bottom-top)).int()

    if with_one_pixel_shift==True: #apply random one pixel shift to the center
        dj = torch.randint(-1, 1, (B,)).to(device)
        di = torch.randint(-1, 1, (B,)).to(device)
        c_x += di
        c_y += dj               

    left = (c_x-0.5*region_width).int()
    right = (c_x+0.5*region_width).int()
    top = (c_y-0.5*region_width).int()
    bottom = (c_y+0.5*region_width).int()    
    
    return torch.stack((left, top, right, bottom),dim=1)

def extract_regions_from_landmarks_batch(images, landmarks, heatmaps, img_size, region_size, with_augmentation, device):
    #input: torch tensors
    #images: [B,1,N,N], landmarks: [B,68,2] 
    #landmarks = landmarks.squeeze(dim=0) # [1,68,2] -> [68,2]
    B = landmarks.size(0)
    region_rois = []
    regions = []
    heatmap_regions = []
    heatmaps = F.interpolate(heatmaps, size=(img_size,img_size), mode='bicubic', align_corners=False)
    all_landmarks = (landmarks +.5) * img_size
    region_start_ids = [[17,36],[22,42],[27],[48]]
    region_stop_ids = [[22,42],[27,48],[36],[68]]
    for start_id_arr, stop_id_arr in zip(region_start_ids,region_stop_ids):
        my_landmarks = []
        my_heatmaps = []
        for start_id, stop_id in zip(start_id_arr, stop_id_arr):
            my_landmarks.append(all_landmarks[:,start_id:stop_id,:]) 
            my_heatmaps.append(heatmaps[:,start_id:stop_id,:,:])
        #landmark list to arr
        my_landmarks = torch.cat(my_landmarks, dim=1)
        my_heatmaps = torch.cat(my_heatmaps, dim=1)
        #get roi: left, top, right, bottom
        if with_augmentation:
            offset_ratio = torch.distributions.uniform.Uniform(0.25,0.5).sample([B]).to(device)
            with_one_pixel_shift=True
        else:
            offset_ratio = 0.25 * torch.ones((B), device=device)
            with_one_pixel_shift=False        
        roi_batch = defineRegionBatch(my_landmarks, offset_ratio, with_one_pixel_shift, device)
        region_rois.append(roi_batch)
        #crop roi
        region_batch = []
        region_heat_batch = []
        for b,roi in enumerate(roi_batch):
            my_roi = roi.clone()
            image = images[b]
            heatmap = my_heatmaps[b]
            offset_min = my_roi.min()
            offset_max = my_roi.max() - images.size(-1)
            if offset_min < 0:
                my_roi = my_roi - offset_min
                image = F.pad(image, (-offset_min, 0, -offset_min, 0)) #(padding_left,padding_right, padding_top,padding_bottom) 
                heatmap = F.pad(heatmap, (-offset_min, 0, -offset_min, 0))
            if offset_max > 0:
                image = F.pad(image, (0, offset_max, 0, offset_max)) 
                heatmap = F.pad(heatmap, (0, offset_max, 0, offset_max)) 
            left, top, right, bottom = my_roi
            region = image[:,top:bottom, left:right]  
            region_heatmap = heatmap[:,top:bottom, left:right] 
            #resize to region size
            region = F.interpolate(region.unsqueeze(0), size=(region_size,region_size), mode='bicubic', align_corners=False)
            region_batch.append(region)
            region_heatmap = F.interpolate(region_heatmap.unsqueeze(0), size=(region_size,region_size), mode='bicubic', align_corners=False)
            region_heat_batch.append(region_heatmap)
        
        region_batch = torch.cat(region_batch, dim=0)
        regions.append(region_batch)
        region_heat_batch = torch.cat(region_heat_batch, dim=0)
        heatmap_regions.append(region_heat_batch)
    return regions, heatmap_regions, region_rois
        
def resample_landmarks_from_regions_batch(images, global_landmarks, region_landmarks, region_rois, device):
    #input: torch tensors
    #image: [B,1,N,N], global_landmarks: [B,68,2], region_landmarks: 4 x [B,N_i,2], region rois: 4 x B x [4]
    img_size = images.size(2)
    landmarks = torch.zeros(global_landmarks.size(), dtype=torch.float, device=device)
    region_start_ids = [[17,36],[22,42],[27],[48]]
    region_stop_ids = [[22,42],[27,48],[36],[68]]
    for i in range(len(region_landmarks)):
        my_landmarks_batch = region_landmarks[i] #.squeeze()
        roi_batch =  region_rois[i] 
        for b,roi in enumerate(roi_batch):
            my_roi = roi.clone()        
            #image = images[b]
            my_landmarks = my_landmarks_batch[b].clone()
            left, top, right, bottom = my_roi       
            #map landmarks to [0,1]
            my_landmarks += 0.5
            #map landmarks to global patch coords
            region_width = max(right - left, bottom - top)
            my_landmarks *= region_width
            my_landmarks += torch.tensor([[left, top]], dtype=torch.float, device=device)
            #map landmarks to [-.5,.5]
            my_landmarks /= img_size
            my_landmarks -= 0.5
            
            start_id_arr = region_start_ids[i]
            stop_id_arr = region_stop_ids[i]
            my_start_id = 0
            for start_id, stop_id in zip(start_id_arr, stop_id_arr):
                landmarks[b,start_id:stop_id,:] = my_landmarks[my_start_id:my_start_id+stop_id-start_id,:] 
                my_start_id += stop_id-start_id
    #add chin line: idx 0-16
    start,stop = 0,17
    landmarks[:,start:stop,:] = global_landmarks[:,start:stop,:]    
    return landmarks

def read_landmarks_pts_format(file):
    pts = np.loadtxt(file, skiprows=3, max_rows=68)
    pts -= 1 #remove 1 (.pts format uses matlab indexing)
    return pts

def write_landmarks_as_pts_format(pts, out_file):    
    pts += 1 #add 1 (.pts format of menpo uses matlab indexing)    
    header = "version: 1\nn_points: {}\n{{".format(pts.shape[0])
    np.savetxt(
        out_file,
        pts,
        delimiter=" ",
        header=header,
        footer="}",
        fmt="%.3f",
        comments="",
    )
                    
def drawAnnotations(img, landmarks, roi=None, save=False, line_thickness=3, point_radius=5):
    outfile="train_patch_anno.jpg"
    # draw landmarks and bounding box
    
    # Chin_Line: 0-16
    start,stop = 0,17
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],False,(255,0,0),thickness=line_thickness)    

    # Eye_Brow_Left
    start,stop = 17,22
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],False,(255,0,0),thickness=line_thickness) 
    
    # Eye_Brow_Right
    start,stop = 22,27
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],False,(255,0,0),thickness=line_thickness) 
    
    # Nasal_Bridge
    start,stop = 27,31
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],False,(255,0,0),thickness=line_thickness) 

    # Nasal_Wings
    start,stop = 31,36
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],False,(255,0,0),thickness=line_thickness)      
    
    # Eye_Left
    start,stop = 36,42
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0),thickness=line_thickness) 
    
    # Eye_Right
    start,stop = 42,48
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0),thickness=line_thickness)       
    
    # Mouth_Outline
    start,stop = 48,60
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0),thickness=line_thickness) 
    
    # Mouth_Inline
    start,stop = 60,68
    pts =  landmarks[start:stop,:]
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0),thickness=line_thickness)  

    #draw all points:
    for landmark in landmarks:   
        cv2.circle(img,(int(landmark[0]),int(landmark[1])), point_radius, (0,0,255), -1)        
    
    if roi is not None:
        # Bounding_Box
        cv2.rectangle(img,(roi[0],roi[1]),(roi[2],roi[3]),(255,255,0),3)
    
    if save:
        cv2.imwrite(outfile, img)
    return img