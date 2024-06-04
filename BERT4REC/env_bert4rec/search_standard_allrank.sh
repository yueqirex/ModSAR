# this script is used to run bert4rec on standard data,
# after varified reproduciblity against original and gold_loss's bert4rec.
# The difference would be this is standard data using sasrec raw .txt
model="BERT4REC"
dataset_ls=("Beauty" "ml-1m" "Video" "Sports" "Yelp")
maxlen_ls=(50 200 50 50 50)
bs_ls=(256 256 256 256 256) # 150 256

# data_standard
max_steps_ls=(-1 -1 -1 -1 -1)

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
data_version=standard # bert4rec, bert4rec_standard, standard
negs_type=pop # uniform, pop
lr=1e-3

inference_only=False

# small letters denote original data, otherwise its standard data
if [ "$data_version" = "bert4rec" ]; then
	dataset_ls=("beauty" "ml-1m" "video" "sports" "yelp")
fi

if [ "$causal_mask" = "True" ]; then
  cm=1
else
  cm=0
fi

if [ "$task_type" = "ae" ]; then
	dp_ls=(0.2 0.2 0.2 0.2 0.2)
	attn_dp_ls=(0.2 0.2 0.2 0.2 0.2)
	mask_prob_ls=(0.5 0.2 0.5 0.5 0.5)
elif [ "$task_type" = "ar" ]; then
	dp_ls=(0.5 0.2 0.5 0.5 0.5)
	attn_dp_ls=(0.5 0.2 0.5 0.5 0.5)
	mask_prob_ls=(0.5 0.2 0.5 0.5 0.5)
fi



d=4 # 1,3,4,5,6
ulimit -n 4096
for ((i=1;i<=1;i++)); do
	for dp in 0.2; do # 
		for wd in 0.1; do
			for head in ${num_head}; do #
				device=$((d%8))
				d=$((d+1))
				if [ "$inference_only" = "True" ]; then
					ckpt_path=checkpoints/your_check_point.ckpt
				else
					ckpt_path=None
				fi
				screen -S ${model}_${data_version}_negs_${negs_type}_${dataset_ls[i]}_seed${seed}_${task_type}_cm${cm}_head${head}_dp${dp}_wd${wd} \
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
								--optimizer_type=adamw \
								--scheduler=default \
								--lr=${lr} \
								--mask_prob=${mask_prob_ls[i]} \
								--max_position=${maxlen_ls[i]} \
								--max_steps=${max_steps_ls[i]} \
								--max_epochs=1000 \
								--batch_size=${bs_ls[i]} \
								--hidden_size=${hidden_size} \
								--dropout_prob=${dp_ls[i]} \
								--attention_dropout_prob=${attn_dp_ls[i]} \
								--num_hidden_layers=${num_attn_layers} \
								--num_attention_heads=${head} \
								--check_val_every_n_epoch=1 \
								--test_ckpt=${ckpt_path} \
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