## just for the mapping from instance feature to language feature

scenes=("figurines" "teatime" "ramen" "waldo_kitchen")
# 

for index in "${!scenes[@]}"; do

dataset_name=${scenes[$index]}
dataset_path="./output/${dataset_name}/"

python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 16 --decoder_dims 32 64 128 256 256 512 --lr 0.001 --dataset_name ${dataset_name}

done