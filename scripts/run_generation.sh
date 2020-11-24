

CUDA_VISIBLE_DEVICES=1,2,3 \
 python3 ../generate_GeDi.py \
 --gen_length 50 \
 --model_type gpt2 \
 --gen_model_name_or_path gpt2 \
 --disc_weight 30 \
 --rep_penalty_scale 10 \
 --filter_p 0.8 \
 --target_p 0.8 \
 --gen_type "gedi" \
 --repetition_penalty 1.2 \
 --mode "sentiment" \
 --secondary_code "food" \
 --prompt "It was shocking" \
 --penalize_cond \
 --do_sample
