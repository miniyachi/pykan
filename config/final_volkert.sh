export CUDA_VISIBLE_DEVICES=0

id=41166
data_folder=./data
t_seed=0
entity_name=sfn-opts
metric=valid_acc
criteria=final
opts=(sfn) #(sgd adam sfn)
epochs=50 #105
n_trials=1 #10

dataset=volkert

# Loop over opts
for opt in "${opts[@]}"
do
    python run_SFN.py --id $id \
                        --data_folder $data_folder \
                        --t_seed $t_seed \
                        --proj_name final_${dataset}_hsodm \
                        --entity_name $entity_name \
                        --tuning_name tune_${dataset}_${opt} \
                        --metric $metric \
                        --criteria $criteria \
                        --opt $opt \
                        --epochs $epochs \
                        --n_trials $n_trials
done