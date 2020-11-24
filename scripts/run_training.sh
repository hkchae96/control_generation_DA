

export lambda_=0.80
export lr=2e-5


#final GeDi LM checkpoint saved at --output_dir
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 ../adversarial_training.py --task_name SST-2 \
  --overwrite_output_dir \
  --do_eval  \
  --do_train \
  --logit_scale \
  --data_dir ../data/AG-news  \
  --max_seq_length 192 \
  --overwrite_cache \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size  8 \
  --learning_rate $lr  \
  --num_train_epochs 1.0  \
  --output_dir ../topic_GeDi_retrained \
  --model_type gpt2  \
  --discriminator bert \
  --model_name_or_path gpt2-medium \
  --discrim_name_or_path bert  \
  --gen_weight $lambda_ \
  --logging_steps 500 \
  --save_steps 1000 \
  --code_0 false \
  --code_1 true  \
  --mode topic  \
  --control_code politics