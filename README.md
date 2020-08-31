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

#### task_type: POS (PoS-tagging)

##### Eval results 
```
eval_loss = 0.12940883628604277
eval_accuracy_score = 0.9644776620759153
eval_precision = 0.9595684967549553
eval_recall = 0.9593160894344586
eval_f1 = 0.9594422764940589
epoch = 3.0
```

##### Test results
```
eval_loss = 0.10366534093226933
eval_accuracy_score = 0.968596946125036
eval_precision = 0.9643250826736146
eval_recall = 0.965002005615724
eval_f1 = 0.964663425392211
```

#### task_type: DEPREL (Universal Dependency Relations)

##### Eval results 
```
eval_loss = 0.2931467722839033
eval_accuracy_score = 0.9117400067181727
eval_precision = 0.9091454016098647
eval_recall = 0.9070482699700982
eval_f1 = 0.9080956250267288
epoch = 3.0
```

##### Test results
```
eval_loss = 0.2617744687395018
eval_accuracy_score = 0.9213483146067416
eval_precision = 0.9188105252861195
eval_recall = 0.9169269816477938
eval_f1 = 0.91786778716959
```

#### task_type: RELPOS (Relative Position between head and id)

##### Eval results 
```
eval_loss = 0.7054957197585576
eval_accuracy_score = 0.7948438024857238
eval_precision = 0.7775862068965518
eval_recall = 0.7679003494936822
eval_f1 = 0.7727129266423194
epoch = 3.0
```

##### Test results
```
eval_loss = 0.6385571194476769
eval_accuracy_score = 0.8070680879669644
eval_precision = 0.7904142134077058
eval_recall = 0.7806570087737197
eval_f1 = 0.785505312323564
```