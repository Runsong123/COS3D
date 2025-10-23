scenes=("figurines" "ramen" "teatime" "waldo_kitchen")
# 
# 
## 3D grouding
python Grounding_3D.py


#### render the grounding results
for index in "${!scenes[@]}"; do    
    python render_3D_prompt.py -m output/${scenes[$index]}    --iteration 30000 --foundation_model "sam_clip" --method_name "Grouding_3D_mask" --normalize_flag     
done

# 
python compute_lerf_iou_check.py --method  Grouding_3D_mask


