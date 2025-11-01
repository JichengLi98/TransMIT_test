export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

fix_seed=${1:-2025}  # default = 2025 if no argument provided

python -u run.py \
  --fix_seed $fix_seed \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FTD.csv \
  --model_id FTD_mask_0.1 \
  --mask_rate 0.1 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 47 \ 
  --dec_in 47 \
  --c_out 47 \
  --batch_size 100 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.0005

python -u run.py \
  --fix_seed $fix_seed \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FTD.csv \
  --model_id FTD_mask_0.2 \
  --mask_rate 0.2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 47 \ 
  --dec_in 47 \
  --c_out 47 \
  --batch_size 100 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.0005

python -u run.py \
  --fix_seed $fix_seed \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FTD.csv \
  --model_id FTD_mask_0.3 \
  --mask_rate 0.3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 47 \ 
  --dec_in 47 \
  --c_out 47 \
  --batch_size 100 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.0005

python -u run.py \
  --fix_seed $fix_seed \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FTD.csv \
  --model_id FTD_mask_0.4 \
  --mask_rate 0.4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 47 \ 
  --dec_in 47 \
  --c_out 47 \
  --batch_size 100 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.0005

python -u run.py \
  --fix_seed $fix_seed \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FTD.csv \
  --model_id FTD_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 47 \ 
  --dec_in 47 \
  --c_out 47 \
  --batch_size 100 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.0005
  
