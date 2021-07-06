#!/bin/bash
#batch_size=(49154 24577 12289 6245 3000 1500)
batch_size=(196615 98308 49154 24577 12289 6245 3000 1500)
export NUM_Epoch=101

# batch_size=(98308)

for i in ${batch_size[@]};do

  python dgl_partition.py  --num-epochs $NUM_Epoch  \
  --dataset ogbn-products \
  --aggre mean \
  --log-every 120   \
  --eval-every 100 \
  --fan-out 10,25 \
  --num-layers 2  \
  --batch-size ${i}  > test_log/${i}_2_layers_with_mem_$NUM_Epoch_time.log

done

# for i in ${batch_size[@]};do

#   python dgl_mini_batch_train.py  --num-epochs $NUM_Epoch  \
#   --dataset ogbn-products \
#   --aggre mean \
#   --log-every 120   \
#   --eval-every 10 \
#   --fan-out 10 \
#   --num-layers 1  \
#   --batch-size ${i}  > test_log/${i}_mini_with_mem_$NUM_Epoch.log

# done

# for i in ${batch_size[@]};do

#   python dgl_partition.py  --num-epochs $NUM_Epoch  \
#   --dataset ogbn-products \
#   --aggre mean \
#   --log-every 120   \
#   --eval-every 100 \
#   --fan-out 10 \
#   --num-layers 1  \
#   --batch-size ${i}  > test_log/${i}_with_mem_$NUM_Epoch.log

# done

# for i in ${batch_size[@]};do

#   python dgl_partition.py  --num-epochs $NUM_Epoch  \
#   --dataset ogbn-products \
#   --aggre mean \
#   --log-every 120   \
#   --fan-out 10 \
#   --num-layers 1  \
#   --batch-size ${i}  > test_log/${i}_$NUM_Epoch_time.log

# done