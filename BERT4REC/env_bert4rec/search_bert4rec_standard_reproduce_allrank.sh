model="BERT4REC"
dataset_ls=("Beauty" "ml-1m" "Video" "Sports" "Yelp")
maxlen_ls=(50 200 50 50 50)
bs_ls=(256 256 256 256 256)
# equal to 1000 epochs:
max_steps_ls=(2097000 401000 1312000 3312000 1290000)

hidden_size=64
seed=0
num_attn_layers=2
# control ae vs ar
# ae
task_type=ae
causal_mask=False
num_head=2
# ar
# task_type=ar
# causal_mask=True
# num_head=1
# evaluation & data & lr
all_rank=True
data_version=bert4rec_standard_reproduce # bert4rec, bert4rec_standard, sasrec
negs_type=pop # pop

if [ "$all_rank" = "True" ]; then
  lr=1e-3
else
  lr=1e-4
fi

# mask_prob is only used for sasrec-like dataloader
# as bert4rec dataloader already saved masked sequence
mask_prob_ls=(0.5 0.2 0.5 0.5 0.5)
if [ "$all_rank" = "True" ]; then
  attn_dp_ls=(0.2 0.2 0.2 0.2 0.2)
else
  attn_dp_ls=(0.2 0.2 0.2 0.2 0.2)
fi

# small letters denote original data, otherwise its standard data
if [ "$data_version" = "bert4rec" ]; then
	dataset_ls=("beauty" "ml-1m" "video" "sports" "yelp")
fi

if [ "$causal_mask" = "True" ]; then
  cm=1
else
  cm=0
fi

d=5

ulimit -n 4096
for ((i=3;i<=3;i++)); do
	for dp in 0.2; do # 
		for wd in 0.01; do
			for head in ${num_head}; do #
				device=$((d%8))
				d=$((d+1))
				screen -dmS ${model}_${data_version}_negs_${negs_type}_${dataset_ls[i]}_${task_type}_cm${cm}_head${head}_dp${dp}_wd${wd} \
				bash -c "
						eval $(conda shell.bash hook); \
						conda activate lightning; \
						CUDA_VISIBLE_DEVICES=6 \
						python main.py \
								--task_type=${task_type} \
								--causal_mask=${causal_mask} \
								--data_version=${data_version} \
								--negs_type=${negs_type} \
								--use_valid=True \
								--all_ranking=${all_rank} \
								--padding_mode=tail \
								--dataset=${dataset_ls[i]} \
								--optimizer_type=adamw \
								--scheduler=linear_decay \
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
								--check_val_every_n_epoch=1 \
								--accelerator=gpu \
								--devices=1 \
								--seed=${seed} \
								--weight_decay=${wd} \
								--num_warmup_steps=100 \
								--early_stopping=True \
								--patience=30 \
								--weight_init=truncated_normal_resample; \
						exec bash;
						"
			done
		done
	done
done
