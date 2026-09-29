export CUDA_VISIBLE_DEVICES=0

ALG_NAME=GLIME

python run.py \
    --alg_name=$ALG_NAME \
    --hparams_fname="./hparams/${ALG_NAME}/llama3.1-8b.yaml" \
    --ds_name=./data/glime_mquake.jsonl \
    --sequential_edit true