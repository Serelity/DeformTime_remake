export CUDA_VISIBLE_DEVICES=0

model_name=DeformTime

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --pred_len 96 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --batch_size 16 \
  --learning_rate 0.001 \
  --n_heads 4 \
  --n_reshape 24 \
  --layer_dropout 0 \
  --patch_len 6 \
  --kernel 7 \
  --patience 5 \
  --dropout 0 \
  --train_epochs 20 \
  --itr 1
  
