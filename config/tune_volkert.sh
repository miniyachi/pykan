export CUDA_VISIBLE_DEVICES=0

id=41166
data_folder=./data
t_seed=1729
o_seed=0
opts=(sgd adam sfn)
epochs=105
n_trials=30

dataset=volkert

# Loop over opts and add 1 to o_seed so that each opt gets different hyperparameters
for opt in "${opts[@]}"
do
    python tuning.py --id $id \
                     --data_folder $data_folder \
                     --t_seed $t_seed \
                     --o_seed $o_seed \
                     --proj_name tune_${dataset}_${opt} \
                     --opt $opt \
                     --epochs $epochs \
                     --n_trials $n_trials
                   
    o_seed=$((o_seed + 1))
done