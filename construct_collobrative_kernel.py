#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render 
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
from scene.dataset_readers import read_sam_clip_feature
import matplotlib.pyplot as plt
import glob
# from utils.graphics_utils import BasicPointCloud
from plyfile import PlyData, PlyElement

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=3, svd_solver='full')
from PIL import Image
import torchvision


import colorsys

import numpy as np



def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask



def get_contrastive_loss(feature, gt_obj):
    
    
    feature = feature.reshape(16, -1).T
    
    loss_regularization = 100 * ((torch.norm(feature, dim=-1, keepdim=True) - 1.0) ** 2).mean()
    
    # feature =  feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-6).detach()
    gt_obj = gt_obj.reshape(-1)
    label_set = torch.unique(gt_obj)
    wh = feature.shape[0]

    # cluster_ids = torch.unique(gt_obj)
    # choose_ids = []
    # clustersize = 500
    # for cid in cluster_ids:
    #     list_ids = torch.arange(wh)[gt_obj == cid]
    #     rand_idx = torch.randint(0, len(list_ids), [clustersize])
    #     choose_ids.append(list_ids[rand_idx])
    # random_idx = torch.cat(choose_ids)
    
    
    batchsize = 32768
    random_idx = torch.randint(0, wh, [batchsize])
    
    # feature =
    sam_t = gt_obj[random_idx]
    sam_o = feature[random_idx]
    
    # print(sam_t.shape)
    # print(sam_o.shape)
    # exit()
    
    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()
    
    
    # results = {'semantic': out_obj[random_idx, :]}
    # target = {'sam': gt_obj[random_idx]}
    
    
    min_pixnum = 20
    cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number

    u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
    phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]
        u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
        phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

    

    # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
    # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
    # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()
    
    ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]

        dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]

        # if not patch_flag:

        ProtoNCE += -torch.sum(torch.log(
            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
            ))
    

    ProtoNCE = ProtoNCE/cnum
    return ProtoNCE, loss_regularization


def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    features_reshaped =  features_reshaped / (torch.norm(features_reshaped, dim=-1, keepdim=True) + 1e-6).detach()
    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')
    return rgb_array


def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1) # channel维度上做归一化
    pca = sklearn.decomposition.PCA(3, random_state=42) # PCA降维到3维
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy() # 1, h, w, 512 -> h*w/3, 512
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu() # h, w, 3
    return vis_feature

def scale_visualize_saving(scale_map):
    max_scale=torch.argmax(scale_map,dim=0) # 0,1,2
    max_scale=max_scale/2
    return max_scale

def process_scale_map(scale_map):
    scale_maps = [torch.zeros_like(scale_map).cuda() for _ in range(3)]
    for i, sm in enumerate(scale_maps):
        sm[i] = 1
    return scale_maps

def process_feature_map(view, scale_map):
    gt_feature_maps = []
    for sm in process_scale_map(scale_map):
        gt_feature_map, mask = read_sam_clip_feature(view.img_embed.cuda(), view.seg_map.cuda(), sm.cuda(), max_mode=True)
        gt_feature_maps.append(gt_feature_map * mask)
    return gt_feature_maps

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    # print(len(extra_f_names))
    max_sh_degree = 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    num_objects = 16
    objects_dc = np.zeros((xyz.shape[0], num_objects, 1))
    for idx in range(num_objects):
        objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["semantic_"+str(idx)])
    # objects_dc = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
    # print("semantic_feature shape: ", objects_dc.shape)
    # print("semantic_feature of Gaussian[0]: ", objects_dc[0])
    # self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    # self._semantic_feature[0] = self._semantic_feature[0] + .0 

    xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

    return [xyz, features_dc, opacity,  features_rest, scaling, rotation, objects_dc]
    # active_sh_degree = max_sh_degree




def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, speedup, feature_mode, feature_npy, render_mode, scene_name, normalize_flag=False):
    image_gt_list = glob.glob(os.path.join(source_path, 'images', '*.*'))
    image_gt_list.sort()
    orig_img_width, orig_img_height = Image.open(image_gt_list[0]).size
    print("gt image size:", orig_img_height, "," , orig_img_width)
    
    device0 = 'cuda'
    device1 = 'cpu'

    iteration = 30000
    
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    feature_map_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map_npy")
    
    
    
    
    if feature_npy:
        makedirs(feature_map_npy_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True)
    
    

    # feature_CLIP_pair = []
    collected_feature = []
    collected_clip = []
    collected_distance = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx>10:
        #     break
        if feature_npy or feature_mode==False: # change to full resolution
            view.image_width = orig_img_width
            view.image_height = orig_img_height

        render_pkg = render(view, gaussians, pipeline, background, feature_mode=feature_mode, render_mode=render_mode,feature_normalize=normalize_flag) 
        feature_map = render_pkg["render"] # 16,731,989

        feature_map = feature_map/(torch.norm(feature_map, dim=0, keepdim=True) +1e-6)
        

        SAM_level = 3
        # gt_clip_feature = view.img_embed[SAM_level,...]
        sam_seg = view.seg_map[SAM_level,...].int()
        gt_clip_feature = view.img_embed.cuda() ### XX * 512
        gt_clip_feature = gt_clip_feature/(torch.norm(gt_clip_feature, dim=-1, keepdim=True) +1e-6)

        unique_obj = torch.unique(sam_seg)
        
        for obj_id in unique_obj:
            if obj_id < 0:
                continue
            obj_mask = sam_seg == obj_id
            # feature_CLIP_pair

            # print("feature_map[...,obj_mask]", feature_map[...,obj_mask].shape)
            mean_feature = feature_map[...,obj_mask].mean(dim=-1)
            distance = torch.norm(feature_map[...,obj_mask]-mean_feature.reshape(-1,1), dim=0, p=2)
            clip = gt_clip_feature[obj_id]
            # print("clip", clip.shape)

            collected_distance.append(distance)
            # feature_CLIP_pair.append([mean_feature, clip])
            collected_feature.append(mean_feature)
            collected_clip.append(clip)
            # exit()
    collected_feature = torch.stack(collected_feature)
    collected_clip = torch.stack(collected_clip)

    collected_distance = torch.cat(collected_distance,dim=0)

    print("collected_distance.shape", collected_distance.shape)
    print("collected_distance.mean", torch.mean(collected_distance))
    print("collected_distance.median", torch.median(collected_distance))
    torch.save(torch.median(collected_distance), f"{model_path}/mean_distance.pt")
    # exit()

    print("collected_feature", collected_feature.shape)
    print("collected_clip", collected_clip.shape)

    print("collected_feature", collected_feature)
    print("collected_clip", collected_clip)
    # exit()
    feature_CLIP_pair = {
        "feature": collected_feature,
        "clip": collected_clip
    }
    torch.save(feature_CLIP_pair, f"{model_path}/feature_CLIP_pair.pt")
    # return torch.tensor(feature_CLIP_pair)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_mode: bool, feature_npy: bool, render_mode: str, normalize_flag: bool): ###
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # 从.ply文件中加载场景信息

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        scene_name = dataset.source_path.split('/')[-1]
        print("scene_name:", scene_name)
        # exit()
        
        checkpoint = f"{dataset.model_path}/chkpnt30000.pth"

        
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_v1(model_params)




        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dataset.speedup = getattr(dataset, 'speedup', False)

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.speedup, feature_mode, feature_npy, render_mode, scene_name, normalize_flag)

        # if not skip_test:
        #      render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.speedup, feature_mode, feature_npy, render_mode, scene_name, normalize_flag)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--feature_mode', action='store_true', help='use feature replace RGB')
    parser.add_argument("--feature_npy", action='store_true', help='store 16-dim feature map in npy')
    parser.add_argument("--render_mode", default="RGB", type=str) # RGB+ED
    parser.add_argument("--normalize_flag", action="store_true") # RGB+ED
    args = get_combined_args(parser) #从命令行获取参数，并根据得到的model_path解析该目录下cfg_args文件，获取训练参数
    print("Rendering " + args.model_path)

    assert not (args.feature_mode and args.render_mode == "RGB+ED"), "Feature mode does not support depth rendering"
    # Initialize system state (RNG)
    args.eval = False
    safe_state(args.quiet)
    print(args)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_mode, args.feature_npy, args.render_mode, args.normalize_flag) ###cnn