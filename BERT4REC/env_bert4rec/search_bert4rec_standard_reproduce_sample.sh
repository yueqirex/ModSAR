# this script is used for searching bert4rec-like-augmented standard data
# using our repoducible data generation code including dup_factor, sliding_window, etc.
model="BERT4REC"
dataset_ls=("Beauty" "ml-1m" "Video" "Sports" "Yelp")
maxlen_ls=(50 200 50 50 50)
bs_ls=(256 256 256 256 256) # 150 256
# data_bert4rec_standard_reproduce 1000 epochs:
max_steps_ls=(2097000 401000 1312000 3312000 1290000)
# mask_prob is only used for sasrec-like dataloader, 
# as bert4rec dataloader already saved masked sequence
mask_prob_ls=(0.6 0.2 0.5 0.5 0.5) 


hidden_size=64
seed=0
num_attn_layers=2
task_type=ar
causal_mask=True

all_rank=False
data_version=bert4rec_standard_reproduce # bert4rec, bert4rec_standard, sasrec
negs_type=pop # uniform, pop

# small letters denote original data, otherwise its standard data
if [ "$data_version" = "bert4rec" ]; then
	dataset_ls=("beauty" "ml-1m" "video" "sports" "yelp")
fi

if [ "$causal_mask" = "True" ]; then
  cm=1
else
  cm=0
fi
if [ "$all_rank" = "True" ]; then
  attn_dp_ls=(0.5 0.2 0.5 0.5 0.5)
else
  attn_dp_ls=(0.2 0.2 0.2 0.2 0.2)
fi

if [ "$all_rank" = "True" ]; then
  lr=1e-3
else
  lr=1e-4
fi

d=4 # 1,3,4,5,6

ulimit -n 4096
# run for different datasets
for ((i=2;i<=2;i++)); do
	for dp in 0.5; do # 
		for wd in 0.01; do
			for head in 2; do #
				device=$((d%8))
				d=$((d+1))
				screen -dmS ${model}_${data_version}_negs_${negs_type}_${dataset_ls[i]}_${task_type}_cm${cm}_head${head}_dp${dp}_wd${wd} \
				bash -c "
						eval $(conda shell.bash hook); \
						conda activate lightning; \
						CUDA_VISIBLE_DEVICES=${device} \
						python main.py \
								--task_type=${task_type} \
								--causal_mask=${causal_mask} \
								--data_version=${data_version} \
								--negs_type=${negs_type} \
								--use_valid=True \
								--all_ranking=${all_rank} \
								--padding_mode=tail \
								--dataset=${dataset_ls[i]} \
								--lr=${lr} \
								--mask_prob=${mask_prob_ls[i]} \
								--max_position=${maxlen_ls[i]} \
								--max_steps=${max_steps_ls[i]} \
								--max_epochs=${max_steps_ls[i]} \
								--batch_size=${bs_ls[i]} \
								--hidden_size=${hidden_size} \
								--dropout_prob=${dp} \
								--attention_dropout_prob=${attn_dp_ls[i]} \
								--num_hidden_layers=${num_attn_layers} \
								--num_attention_heads=${head} \
								--check_val_every_n_epoch=5 \
								--accelerator=gpu \
								--devices=1 \
								--seed=${seed} \
								--weight_decay=${wd} \
								--num_warmup_steps=100 \
								--early_stopping=True \
								--patience=10 \
								--weight_init=truncated_normal_resample; \
						exec bash;
						"
			done
		done
	done
done