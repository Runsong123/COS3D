# scenes=("figurines" "ramen" "teatime" "waldo_kitchen")
scenes=("figurines")

# 

mkdir log
mkdir output
# stage 1
for index in "${!scenes[@]}"; do

    adjusted_index=$((index))
    adjusted_ip=$((index+6200))   
    CUDA_VISIBLE_DEVICES=$adjusted_index  python train_instance.py -s ../OpenGaussian_v1/data/${scenes[$index]}/  -m output/${scenes[$index]}_again  --original_model_path ../gaussian-splatting/output/${scenes[$index]}_eval  --feature_mode --SAM_level 3  --port ${adjusted_ip} --eval 

done


### stage 2: Instance2language mapping
### we provide two implmentations (Kernel version or the MLPs versoin.)
### for kernel version, the mapping is training free, you can just run collecting the part to direct calculate the mapping function.
scenes=("figurines" "ramen" "teatime" "waldo_kitchen")

# mkdir output
for index in "${!scenes[@]}"; do
    python construct_collobrative_kernel.py -m  output/${scenes[$index]} --foundation_model "sam_clip" --feature_mode --feature_npy --normalize_flag
    
done


### for MLPs version, you need to additionally train MLPs by following script.

cd ./autoencoder && bash ./autoencoder/bash_run.sh

