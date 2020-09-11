# You can download UD 2.6 for all languages
# curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226{/ud-treebanks-v2.6.tgz,/ud-documentation-v2.6.tgz,/ud-tools-v2.6.tgz}

export MAX_LENGTH=200
export BERT_MODEL=bert-base-uncased
export OUTPUT_DIR=transformers-pat-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

TRAIN_FILE=it_isdt-ud-train.conllu
DEV_FILE=it_isdt-ud-dev.conllu
TEST_FILE=it_isdt-ud-test.conllu

if [ -e "$TRAIN_FILE" ] && [ -e "$DEV_FILE" ] && [ -e "$TEST_FILE" ]
then
    echo "Found dataset"
else
    if [ -e UD2.6_it.zip ]
    then
        unzip UD2.6_it.zip -d .
        for FILE in *
        do
            if [[ "$FILE" == *.conllu ]]
            then
                sed '/[0-9]\+-/d' "$FILE" > "${FILE%.*}".txt
                rm "$FILE"
            fi
        done
    else
        echo "Dataset not found"
        exit
    fi
fi

# --task_type: POS DEPREL RELPOS

python3 run_ner.py \
--task_type DEPREL \
--data_dir . \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
