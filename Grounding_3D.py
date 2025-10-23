import numpy as np
import os
import sys
sys.path.append('/research/d1/gds/rszhu22/Gaussian_Segmentation/gaussian-grouping_replica') 

import sys
from PIL import Image
import glob
import colorsys
import cv2
import numpy as np
from utils.graphics_utils import BasicPointCloud
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
from torch import nn
import shutil

# import pytorch3d
import pytorch3d.ops as ops

import time



import json

from eval.openclip_encoder import OpenCLIPNetwork
import torch.nn.functional as F
# from gaussian_renderer import render, render_select_segment
import matplotlib.pyplot as plt




## I want to keep the centroid is always the same, as the input pts is the always the same for same machine
# Randomly initialize 300 colors for visualizing the SAM mask. [OpenGaussian_v1]
np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)

def construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    for i in range(objects_dc.shape[1]*objects_dc.shape[2]):
        l.append('obj_dc_{}'.format(i))
    return l

def save_ply(xyz, features_dc, opacity,  features_rest, scaling, rotation, objects_dc, path):
    # mkdir_p(os.path.dirname(path))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    obj_dc = objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

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

    # opacity = vertices['opacity']
    # filter_indice = opacity > 0.5

    xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

    return [xyz, features_dc, opacity,  features_rest, scaling, rotation, objects_dc]
    # active_sh_degree = max_sh_degree


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # objects = vertices['objects'].T
    
    num_objects = 16
    objects_dc = np.zeros((positions.shape[0], num_objects, 1))
    for idx in range(num_objects):
        objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])
            
    print(objects_dc.shape)
    return positions, objects_dc





# 
# 
def feature2CLIP_3D(feature_obj, CLIP_pair):
    anchor_feature = CLIP_pair['feature']
    anchor_clip = CLIP_pair['clip']


    #

    ### 
    chunksize = 10000
    for i in range(0, feature_obj.shape[0], chunksize):
        similarity = torch.argmax(torch.matmul(feature_obj[i:i+chunksize], anchor_feature.T), dim=-1)
        if i == 0:
            similarity_all = similarity
        else:
            similarity_all = torch.cat((similarity_all, similarity), dim=0)
    print("similarity_all", similarity_all.shape)
    lifted_clip = anchor_clip[similarity_all]
    return lifted_clip


@torch.no_grad()
def Instance2Language_kernel(feature_obj, CLIP_pair):
    anchor_feature = CLIP_pair['feature']
    anchor_clip = CLIP_pair['clip']
    print("anchor_clip.shape", anchor_clip.shape)

    # anchor_feature = anchor_feature/
    # select_feature 
    # similarity = torch.argmax(torch.matmul(feature_obj, anchor_feature.T), dim=-1)

    ### please calculte the similarity between feature_obj and anchor_feature using chunked way
    chunksize = 1000
    k = 25
    sigma = 0.1
    for i in range(0, feature_obj.shape[0], chunksize):

        # 
        # distances = torch.matmul([x_query], self.X_train, metric='euclidean')[0]
        distances = 1 - torch.matmul(feature_obj[i:i+chunksize], anchor_feature.T)
        print("distances.shape", distances.shape)

        # 
        # nearest_idx = torch.argsort(distances, dim=-1)[...,:k]
        nearest_distances, nearest_idx = torch.topk(distances, k, largest=False, dim=1)  # (M, k)
        # nearest_X = self.X_train[nearest_idx]
        # nearest_Y = self.Y_train[nearest_idx]
        nearst_CLIP = anchor_clip[nearest_idx]
        # nearest_distances = distances[nearest_idx]

        print("nearest_distances.shape", nearest_distances.shape)

        # 
        # if self.weight_type == 'inverse_distance':
        #     weights = 1 / (nearest_distances + 1e-5)  # 避免除零
        # elif self.weight_type == 'gaussian':
        weights = torch.exp(-nearest_distances**2 / (2 * sigma**2))
        # else:
        #     raise ValueError("Invalid weight_type. Choose 'inverse_distance' or 'gaussian'.")

        # 
        weights /= weights.sum()

        print("nearst_CLIP",nearst_CLIP.shape)
        print("weights[:, None]",weights.shape)

        # 
        new_CLIP = torch.sum(nearst_CLIP * weights[..., None], dim=1)
        new_CLIP = new_CLIP/torch.norm(new_CLIP,dim=-1, keepdim=True)
        new_CLIP = new_CLIP.cpu()
        # similarity = torch.argmax(torch.matmul(feature_obj[i:i+chunksize], anchor_feature.T), dim=-1)
        if i == 0:
            lifted_clip = new_CLIP
        else:
            lifted_clip = torch.cat((lifted_clip, new_CLIP), dim=0)
    # print("similarity_all", similarity_all.shape)
    # lifted_clip = anchor_clip[similarity_all]
    return lifted_clip.cuda()


@torch.no_grad()
def smooth_feature_in_3D(xyz, clip_feature):
    ### space smooth
    smoothed_dim = 512
    K = 10
    nearest_k_idx = ops.knn_points(
        xyz.unsqueeze(0),
        xyz.unsqueeze(0),
        K=K,
    ).idx.squeeze()
    # self.feature_smooth_map = {"K":K, "m":nearest_k_idx}
    cur_features = clip_feature
    # cur_features = self._point_features
    
    
    chunksize = 10000
    
    for i in range(0, clip_feature.shape[0], chunksize):
        
        cur_features[i:i+chunksize, :smoothed_dim] = cur_features[nearest_k_idx[i:i+chunksize], :smoothed_dim].mean(dim = 1).detach()
        
    # cur_features[:, :smoothed_dim] = cur_features[nearest_k_idx, :smoothed_dim].mean(dim = 1).detach()
    cur_features = cur_features/cur_features.norm(dim=1, keepdim=True)

    return cur_features



def greedy_feature_selection_with_score_filter(scores, features, opacity, sim_thresh_original,
                                               score_thresh=0.5, 
                                               mean_score_thresh=0.3):
    """
    Greedy selection based on score and feature similarity, with mean score filtering.

    Args:
        scores (Tensor): [N, 1] scores in [0,1]
        features (Tensor): [N, 16] feature vectors
        score_thresh (float): threshold to pick initial point
        sim_thresh (float): cosine similarity threshold for collecting neighbors
        mean_score_thresh (float): minimum mean score to accept a selected group

    Returns:
        collected_indices (List[int]): all accepted selected point indices
    """
    # original_

    scores = scores.squeeze()  # [N]
    features = F.normalize(features, dim=1)  # Normalize for cosine similarity
    N = scores.shape[0]

    processed = torch.zeros(N, dtype=torch.bool).cuda()
    collected_indices = []

    #
    #
    sim_thresh_list = [1 - 0.4 * sim_thresh_original, 1 - 0.3 * sim_thresh_original, 1 - 0.2 * sim_thresh_original, 1 - 0.1 * sim_thresh_original,]
    # 
    print("len(sim_thresh_list)", len(sim_thresh_list))
    # exit()
    while True:
        # Candidates: unprocessed and score > threshold
        candidates = (~processed) & (scores > score_thresh) & (opacity>0.1)
        if candidates.sum() == 0:
            break

        candidate_indices = candidates.nonzero(as_tuple=False).squeeze()
        if candidate_indices.ndim == 0:
            candidate_indices = candidate_indices.unsqueeze(0)

        max_score_idx = candidate_indices[(scores)[candidate_indices].argmax()].item()

        picked_feature = features[max_score_idx].unsqueeze(0)  # [1, 16]
        sims = F.cosine_similarity(picked_feature, features, dim=1)  # [N]

        for sim_thresh in sim_thresh_list:
            similar_mask = sims > sim_thresh  # [N]

            selected_mask = similar_mask & (~processed)
            selected_indices = selected_mask.nonzero(as_tuple=False).squeeze()

            # Ensure selected_indices is always 1D
            if selected_indices.ndim == 0:
                selected_indices = selected_indices.unsqueeze(0)

            if selected_indices.numel() == 0:
                # No valid similar points
                processed[max_score_idx] = True
                break

            # ratio = (scores[selected_indices]>score_thresh).float().mean().item()
            mean_score = (scores*opacity)[selected_indices].sum()/(opacity[selected_indices].sum())
            # import ipdb
            # ipdb.set_trace()

            if mean_score >= score_thresh:
                collected_indices.extend(selected_indices.tolist())
                break
        processed[selected_indices] = True
        # Mark all selected as processed regardless of acceptance
        

    return collected_indices




scenes=("figurines", "ramen", "teatime", "waldo_kitchen")
# scenes=("figurines", )

# 

for scene_name in scenes:

    
    exp_dir = f"./output/{scene_name}/"
    CLIP_pair = torch.load(f"{exp_dir}/feature_CLIP_pair.pt")


    similairty_file = f"{exp_dir}/mean_distance.pt"

    mean_similarity_threshold = torch.load(similairty_file)
    

    ### 
    pcd_path = f"{exp_dir}/point_cloud/iteration_30000/point_cloud.ply"

    res_list = load_ply(pcd_path)
    original_feature_obj = res_list[-1]
    xyz = res_list[0]
    original_opacity = res_list[2].reshape(-1)

    print("xyz.shape", xyz.shape)
    print("feature_obj.shape", original_feature_obj.shape)
    print("opacity.shape", original_opacity.shape)
    # exit()
    print("torch.min(original_opacity)", torch.min(original_opacity))
    # exit()
    # filter_mask = original_opacity>-1

    xyz = xyz#[filter_mask]
    feature_obj = original_feature_obj#[filter_mask]
    opacity = original_opacity#[filter_mask]
    opacity = torch.sigmoid(opacity)
    
    xyz_np = xyz.detach().cpu().numpy().reshape(-1,3)
    


    feature_obj = feature_obj.reshape(-1, 16).float()
    feature_obj = feature_obj/feature_obj.norm(dim=1, keepdim=True)
    print("feature_obj.shape", feature_obj.shape)
    # exit()

    if os.path.exists(f"{exp_dir}/matching_CLIP_3D.pt"):
        matching_CLIP_3D = torch.load(f"{exp_dir}/matching_CLIP_3D.pt")
        print("matching_CLIP_3D.shape", matching_CLIP_3D.shape)


    else:

        matching_CLIP_3D = Instance2Language_kernel(feature_obj, CLIP_pair)
        # matching_CLIP_3D = torch.randn(feature_obj.shape[0], 512).cuda()    

        print("matching_CLIP_3D.shape", matching_CLIP_3D.shape)


        matching_CLIP_3D = smooth_feature_in_3D(xyz, matching_CLIP_3D)
        print("after_filter_matching_CLIP_3D.shape", matching_CLIP_3D.shape)

        torch.save(matching_CLIP_3D, f"{exp_dir}/matching_CLIP_3D.pt")






    scene_texts = {
        "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
                'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
                'pour-over vessel', 'knife', 'yellow desk'],
        "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
        "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
        # "figurines": ['red apple'],
        # "teatime": [ 'tea in a glass', 'apple']
        "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
                'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                'bag of cookies']
    }
    # note: query text
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = OpenCLIPNetwork(device)
    target_text = scene_texts[scene_name]
    
    # 
    
    ##
    clip_model.set_positives(target_text)    
    # 
    leaf_lang_feat = matching_CLIP_3D.unsqueeze(0)
    valid_map_3D = clip_model.get_max_across_3D(leaf_lang_feat)
    print("target_text length", len(target_text))
    print(valid_map_3D.shape)

    
    target_text = scene_texts[scene_name]

    for t_i, _ in enumerate(target_text):

        text_item = target_text[t_i]
        print(f"rendering the {t_i+1}-th query of {len(target_text)} texts: {target_text[t_i]}")

        cosine_similarity = valid_map_3D[0, t_i,:,0]
        # 
        output = cosine_similarity - torch.min(cosine_similarity)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        cosine_similarity = output



        

        # 

        CLIP_score = cosine_similarity
        collected_indice = greedy_feature_selection_with_score_filter(CLIP_score, feature_obj, opacity, mean_similarity_threshold)
        collected_indice = torch.tensor(collected_indice)
        Gaussian_mask_final = torch.zeros_like(CLIP_score).cuda().bool()
        Gaussian_mask_final[collected_indice] = True
        


        # 
        print("Gaussian_mask_final.shape", Gaussian_mask_final.shape)
        mask_folder_name = "Grouding_3D_mask"
        mask_exp_dir = f"{exp_dir}/{mask_folder_name}/"
        os.makedirs(mask_exp_dir, exist_ok=True)
        np.save(f"{mask_exp_dir}/{text_item}.npy", Gaussian_mask_final.cpu().numpy())
        # 
        
