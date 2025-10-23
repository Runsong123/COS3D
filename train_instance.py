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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l1_loss_map, Scale_balance_loss, scale_regulation_loss, scale_region_regulation_loss, get_trained_seg
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from scene.dataset_readers import read_sam_clip_feature
from segment_anything import sam_model_registry
from preprocess import OpenCLIPNetworkConfig, OpenCLIPNetwork

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=3, svd_solver='full')
from PIL import Image
import torchvision


import colorsys

import numpy as np
## I want to keep the centroid is always the same, as the input pts is the always the same for same machine
# Randomly initialize 300 colors for visualizing the SAM mask. [OpenGaussian_v1]
# random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)


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

def create_scale_map(single_scale, feature_map_shape):
    scale_values = {
        "s": [1, 0, 0],
        "m": [0, 1, 0],
        "l": [0, 0, 1],
        "mix": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
    assert single_scale in scale_values, "Invalid scale value"
    scale_map = torch.tensor(scale_values[single_scale], dtype=torch.float32, device='cuda')
    return scale_map.unsqueeze(-1).unsqueeze(-1).repeat(1, feature_map_shape[1], feature_map_shape[2])


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scale_balance_iteration, scale_regulation_iteration, render_novel_view_iteration, novel_view_interval, feature_mode, single_scale, SAM_level, GS_original_path):
    device0='cuda'
    device1='cpu'
        
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    #
    viewpoint_stack = scene.getTrainCameras()
    camnum_orig=len(viewpoint_stack)
    viewpoint_cam0 = viewpoint_stack[0] 
    feature_out_dim = viewpoint_cam0.img_embed.shape[1]
    render_h,render_w = viewpoint_cam0.image_height, viewpoint_cam0.image_width
    print("render img with H,W:",render_h,",",render_w)
    
    
    gaussians.training_setup(opt)
    
    original_h, original_w = viewpoint_cam0.semantic_feature_height, viewpoint_cam0.semantic_feature_width
    
    # 
    # exit()
    if True: # continue from checkpoint
        # (model_params, first_iter) = torch.load(checkpoint)
        gaussians = GaussianModel(dataset.sh_degree)

        # new_path = dataset.model_path
        # print("dataset.model_path", dataset.model_path)
        # exit()
        dataset.original_model_path = GS_original_path
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False) # 从.ply文件中加载场景信息
        # print("gaussians", gaussians)
        # print("hello")
        gaussians.training_setup(opt)

        # print(len(model_params))
        # print(len(model_params) == 12)
        # exit()
        if feature_mode: 
            first_iter = 0
            # print(model_params)
        # else: # feature field
            # load feature decoder ckpt
            
        # gaussians.restore(model_params, opt)
        print("number of gaussians",gaussians._xyz.shape)
    
    print("SAM_level", SAM_level)
    # set other parameters    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True) 
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
                
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()) #将net_image处理并通过memoryview提供一个直接访问这些数据的接口 clamp()将net_image的值截断至[0,1]
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy() 
        select_idx = randint(0, len(viewpoint_stack)-1) 
        viewpoint_cam = viewpoint_stack.pop(select_idx) 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, feature_mode=feature_mode, feature_normalize=True) 
        feature_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        #


        SAM_mask = viewpoint_cam.seg_map[SAM_level,:,:]
        #
        
        contrastive_loss, reg_loss = get_contrastive_loss(feature_map, SAM_mask)
        loss = (contrastive_loss + reg_loss) * 1e-3

        
    
        loss.backward()
        iter_end.record()
        #

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 500 ==0:
                rgb_mask = feature_to_rgb(feature_map)
                
                os.makedirs(os.path.join(scene.model_path, "save_img"), exist_ok = True)
                # 
                Image.fromarray(rgb_mask).save(os.path.join(scene.model_path, "save_img/PCA_Feature_iteration_{0:05d}.png".format(iteration)))
                SAM_mask_vis = visualize_obj(SAM_mask.cpu().numpy().astype(np.uint8))
                
                Image.fromarray(SAM_mask_vis).save(os.path.join(scene.model_path, "save_img/SAM_{0:05d}.png".format(iteration)))
                
                

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max memory used: {mem:.2f} GB")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and not feature_mode:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, feature_map.shape[2], feature_map.shape[1])
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # 增删高斯
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() 
                    
            # Optimizer step
            if iteration < opt.iterations:
                Debug_flag = False
                if Debug_flag:
                    import ipdb
                    ipdb.set_trace()
                # exit()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")                
                

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_feature, feature_reionvar_loss, scale_s, sclae_m, scale_l, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/feature_reionvar_loss', feature_reionvar_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('scale_patchs/subpart',scale_s.item(), iteration)
        tb_writer.add_scalar('scale_patchs/part',sclae_m.item(), iteration)
        tb_writer.add_scalar('scale_patchs/whole',scale_l.item(), iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser) 
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--SAM_level', type=int, default=3)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 6000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--scale_balance_iteration', type=int, default=1)
    parser.add_argument('--scale_regulation_iteration', type=int, default=15001)
    parser.add_argument('--render_novel_view_iteration',type=int, default=99999)
    parser.add_argument('--novel_view_interval',type=int,default=150)
    parser.add_argument('--feature_mode', action='store_true', help='use feature replace RGB')
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument("--novel_view", action='store_true')
    parser.add_argument("--single_scale",type=str, choices=['s', 'm', 'l', 'mix'], default = None) # s | m | l
    args = parser.parse_args(sys.argv[1:]) 
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    scene_name = args.source_path.split("/")[-1]
    if scene_name == "":
        scene_name = args.source_path.split("/")[-2]
    # print("Scene name: ", scene_name)
    # exit()

    GS_original_path = f"/research/d1/gds/rszhu22/Gaussian_vocabulary/gaussian-splatting/output/{scene_name}_eval/"

    # Initialize SAM & CLIP model
    if args.novel_view:
        CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path).to('cuda')
    else:
        CLIP_model = None
        sam = None
        
    # empty cache
    torch.cuda.empty_cache()
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.scale_balance_iteration, args.scale_regulation_iteration,args.render_novel_view_iteration,args.novel_view_interval,args.feature_mode,args.single_scale, args.SAM_level, GS_original_path)

    # All done
    print("\nTraining complete.")
