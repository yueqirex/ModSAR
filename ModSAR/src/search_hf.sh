# ===== tune =====
dataset_ls=(beauty sports video yelp ml-1m) #
aspect=llama # aspect here means model (albert, llama, transfo_xl)
task_type=ar # ae
gpu=0 # $((gpu % 8))
for ((i=4;i<=4;i++)); do
    screen -S ${aspect}_${dataset_ls[i]}_${task_type} \
    bash -c "
            eval $(conda shell.bash hook); \
            conda activate lightning; \
            CUDA_VISIBLE_DEVICES=0 \
            python tune.py \
            --config configs/hugging_face/${aspect}/${dataset_ls[i]}_config_tune_${task_type}.yaml \
            --run.name=${aspect}_${task_type}_${dataset_ls[i]} \
            -- \
            --config configs/hugging_face/${aspect}/${dataset_ls[i]}_config_run.yaml \
            --model.model_type=${aspect}; \
            exec bash;
            "
    gpu=$((gpu + 1))
done



# ===== check errored trials =====
# bash -c "
#     eval $(conda shell.bash hook); \
#     conda activate lightning; \
#     python evaluation.py --results_dir=output_hf --mode=detect_error;
#     "



# ===== retrive best trial dir + export best config =====
# aspect_ls=("albert", "llama" "transfo_xl")
# task_ls=("ar" "ae")
# ds_ls=("beauty" "sports" "video" "yelp" "ml-1m")
# for asp in "${aspect_ls[@]}"; do
#     for task in "${task_ls[@]}"; do
#         for ds in "${ds_ls[@]}"; do
#             bash -c "
#                 eval $(conda shell.bash hook); \
#                 conda activate lightning; \
#                 CUDA_VISIBLE_DEVICES=0 \
#                 python evaluation.py --mode=export_best_config --results_dir=output_hf --exp_name=${asp}_${task}_${ds} --device=gpu --seed=0;
#                 "
#         done
#     done
# done



# ===== rerun best config =====
# aspect_ls=("albert" "llama" "transfo_xl")
# task_ls=("ae" "ar")
# ds_ls=("beauty" "sports" "video" "yelp" "ml-1m")
# i=2
# for asp in "${aspect_ls[@]}"; do
#     for task in "${task_ls[@]}"; do
#         for ds in "${ds_ls[@]}"; do
#             screen -dmS rerun_${asp}_${task}_${ds} \
#             bash -c "
#                 eval $(conda shell.bash hook); \
#                 conda activate lightning; \
#                 CUDA_VISIBLE_DEVICES=$((i % 8)) \
#                 python main.py fit --config output_hf/${asp}_${task}_${ds}/best_trial/best_config.yaml; \
#                 exec bash;
#                 "
#             i=$((i + 1))
#         done
#     done
# done



# ===== compare the best results for all =====
# python evaluation.py --results_dir=output_hf --mode=compare --with_config=False --succinct=True;



# ===== get run time =====
# bash -c "
#     eval $(conda shell.bash hook); \
#     conda activate lightning; \
#     python evaluation.py --results_dir=output_hf --mode=get_run_time;
#     "