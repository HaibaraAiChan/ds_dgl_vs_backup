#!/bin/bash
export RANK=0
export WORLD_SIZE=1
# batch_size=(24577 12289 6245 3000 1500)
# batch_size=(196615 98308 49154 24577 12289 6245 3000 1500)
batch_size=(196615 )
for i in ${batch_size[@]};do

  python dgl_mini_batch_train.py\
  --num-epochs 11\
  --aggre mean\
  --batch-size $i\
  --log-every 120 \
  --num-layers 1\
  --batch-size ${batch_size[i]}\
  > test_log/${batch_size[i]}_101.log
  
done













# for i in ${batch_size[@]};do

#   # CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  train_serial.py --batch-size $i --deepspeed_config ds_config.json > serial_log/${i}_101.log
#   CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  ds_pseudo_mini_batch_train.py --aggre lstm --batch-size $i --log-every 120 --deepspeed_config ds_config.json > pseudo_log_1/${i}_101.log 
# done

# for i in ${batch_size[@]};do

#   CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  ds_pseudo_mini_batch_train.py --aggre lstm --batch-size $i --log-every 120 --deepspeed_config ds_config.json > pseudo_log_2/${i}_101.log  
# done

# for i in ${batch_size[@]};do

#   CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  ds_mini_batch_train.py --aggre lstm --batch-size $i --log-every 120 --deepspeed_config ds_config.json > basic_log_1/${i}_101.log 
# done

# for i in ${batch_size[@]};do

#   CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  ds_mini_batch_train.py --aggre lstm --batch-size $i --log-every 120 --deepspeed_config ds_config.json > basic_log_2/${i}_101.log 
# done