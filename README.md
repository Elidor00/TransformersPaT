# TransformersPaT

Token classification based on [Name Entity Recognition](https://github.com/huggingface/transformers/tree/master/examples/token-classification) script from [transformers](https://github.com/huggingface/transformers).

### Environment set-up
```
conda create -n transformers-pat python=3.6.9
conda activate transformers-pat
pip install transformers==3.0.2 seqeval==0.0.12 conllu==4.0 torch==1.4.0 torchvision==0.5.0 pytorch-pretrained-bert==0.6.2 networkx==2.4
```

### Data

Data can be obtained from the [Universal Dependencies](https://universaldependencies.org/#download) download page.

In this repo i'm using italian-isdt UD version 2.6.

### Tasks

For now the tasks available are only NER and PoS-tagging. 
Thanks to this [PR](https://github.com/huggingface/transformers/pull/6457) is now possible to add new type of tasks in ```tasks.py```.

### Run

You can train and predict, with Pytorch version, just by running the script ```run_ner.sh```.

The variables are:

```
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
```

The script parameters are:

```
python3 run_ner.py \
--task_type POS \
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

```

### Results

#### PoS-tagging

##### Eval results 
```
eval_loss = 0.12719589757772398
eval_accuracy_score = 0.9656533422908968
eval_precision = 0.9609080550442634
eval_recall = 0.9612450679526523
eval_f1 = 0.9610765319540634
epoch = 3.0
```

##### Running prediction
```
eval_loss = 0.10841419372219042
eval_accuracy_score = 0.9673485066743494
eval_precision = 0.962717979555021
eval_recall = 0.9632972322503008
eval_f1 = 0.963007518796992
```