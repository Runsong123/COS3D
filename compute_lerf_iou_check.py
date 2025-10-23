import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union




def get_filter_set(scene_name, frame_name, label_path):
# def get_label_single_frame(scene_name, frame_name):
    label_list = set()
    label_path = f"{label_path}/{frame_name}"
    for label_file in os.listdir(label_path):
        if label_file.endswith(".jpg") or label_file.endswith(".png"):
            label = label_file.split(".")[0]
            label_list.add(label)
    return label_list

def evalute(gt_base, pred_base, scene_name, filter_set=None, part=False):
    
    
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    
    frame_names = scene_gt_frames[scene_name]

    ious = []
    for frame in frame_names:
        print("frame:", frame)
        gt_floder = os.path.join(gt_base, frame)
        file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg') or f.endswith('.png')]
        for file_name in file_names:
            base_name = os.path.splitext(file_name)[0]
            gt_obj_path = os.path.join(gt_floder, file_name)
            pred_obj_path = os.path.join(pred_base, frame + "_" + base_name + '.png')
            if not os.path.exists(pred_obj_path):
                print(f"Missing pred file for {file_name}, skipping...")
                print(f"IoU for {file_name}: 0")
                ious.append(0.0)
                continue
            mask_gt = load_image_as_binary(gt_obj_path)
            mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
            iou = calculate_iou(mask_gt, mask_pred)
            ious.append(iou)
            print(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}")
    
    # Acc.
    total_count = len(ious)
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    
    # 
    average_iou = np.mean(ious)
    acc_025 = count_iou_025/total_count
    acc_050 = count_iou_05/total_count

    print(f"Average IoU: {average_iou:.4f}")
    print(f"Acc@0.25: {acc_025:.4f}")
    print(f"Acc@0.5: {acc_050:.4f}")
    
    
    return average_iou, acc_025, acc_050

if __name__ == "__main__":
    parser = ArgumentParser("Compute LeRF IoU")
    parser.add_argument("--method", type=str,
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--visible_in_query_frame", action='store_true',
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--part", action='store_true',
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    args = parser.parse_args()

    method = args.method
    part = args.part
    # 
    scenes=("figurines",  "teatime","ramen", "waldo_kitchen")
    # 

    # 
    visible_in_query_frame = args.visible_in_query_frame

    if True:
        
        csv_results_path = f"results/results_COS3D/opengaussian_pts.csv"
        latex_results_path = f"results/results_COS3D/latex_v1.txt"

        os.makedirs(os.path.dirname(csv_results_path), exist_ok=True)
        evaluation_txt_file = open(csv_results_path, "w")
        evaluation_txt_file.write(f"scene_name, mIoU, Acc@0.25, Acc@0.5\n")
        # evaluation_txt_file.flush()
        
        
        
        ## record the results
        iou_list = []
        acc_25_list = []
        acc_50_list = []
        for scen_item in scenes:
        
            
            # TODO: change
            
            path_gt = f"/research/d1/gds/rszhu22/Gaussian_vocabulary/data/lerf_ovs/label/{scen_item}/gt"
            output_file_name = f"{scen_item}"            
            path_pred = f"output/{output_file_name}/text2obj_prompt_{method}/ours_None/renders_cluster_silhouette"

            filter_set = None
            iou, acc_025, acc_050 = evalute(path_gt, path_pred, scen_item, filter_set, part)
            iou_list.append(iou)
            acc_25_list.append(acc_025)
            acc_50_list.append(acc_050)
        
        for i in range(4):
            print(f"{scenes[i]}, {iou_list[i]*100:.2f}, {acc_25_list[i]*100:.2f}, {acc_50_list[i]*100:.2f}\n")
            evaluation_txt_file.write(f"{scenes[i]}, {iou_list[i]*100:.2f}, {acc_25_list[i]*100:.2f}, {acc_50_list[i]*100:.2f}\n")
            # 

        average_iou = np.mean(iou_list) * 100
        average_acc_025 = np.mean(acc_25_list) * 100
        average_acc_050 = np.mean(acc_50_list) * 100
        evaluation_txt_file.write(f"Average, {average_iou:.2f}, {average_acc_025:.2f}, {average_acc_050:.2f}\n")
        evaluation_txt_file.flush()
        
        
        ### 
        os.makedirs(os.path.dirname(latex_results_path), exist_ok=True)
        latex_txt_file = open(latex_results_path, "w")
        latex_txt_file.write(f"COS3D res &   {average_iou:.2f} & {average_acc_025:.2f} & {iou_list[0]*100:.2f} & {acc_25_list[0]*100:.2f} & {iou_list[1]*100:.2f} & {acc_25_list[1]*100:.2f}   & {iou_list[2]*100:.2f} & {acc_25_list[2]*100:.2f} & {iou_list[3]*100:.2f} & {acc_25_list[3]*100:.2f}  \\\\ \n")        
        latex_txt_file.flush()
    
    
    
    
    
    
    
        