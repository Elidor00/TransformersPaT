# TransformersPaT

Token classification based on [Name Entity Recognition](https://github.com/huggingface/transformers/tree/master/examples/token-classification) script from [transformers](https://github.com/huggingface/transformers).

### Environment set-up
```
conda create -n transformers-pat python=3.6.9
conda activate transformers-pat
pip install -r requirements.txt
```

### Data

Data can be obtained from the [Universal Dependencies](https://universaldependencies.org/#download) download page.

In this repo i'm using italian-isdt UD version 2.6.

### Tasks

For now the tasks available are only NER and PoS-tagging. 
Thanks to this [PR](https://github.com/huggingface/transformers/pull/6457) is now possible to add new type of tasks in ```tasks.py```.

Update:
- Create DEPREL-tagging and RELPOS tasks

(RELPOS task is inspired by this [article](https://www.aclweb.org/anthology/2020.lrec-1.643/))

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

### Tasks Results
#### Bert_model: bert-base-uncased

<details>
  <summary>Results with bert-base-uncased model</summary>
  
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
eval_loss = 0.2954632102603644
eval_accuracy_score = 0.9126637554585153
eval_precision = 0.9099708754497173
eval_recall = 0.907560871422469
eval_f1 = 0.9087642756319775
epoch = 3.0
```

##### Test results
```
eval_loss = 0.2616970073737082
eval_accuracy_score = 0.9204840103716508
eval_precision = 0.9175308158873019
eval_recall = 0.9155603279968763
eval_f1 = 0.9165445128505815
```

#### task_type: RELPOS (Relative Position between head and id)

##### Eval results 
```
eval_loss = 0.7067813938352424
eval_accuracy_score = 0.7946758481692979
eval_precision = 0.7764951826940556
eval_recall = 0.7655703916121517
eval_f1 = 0.7709940887144081
epoch = 3.0
```

##### Test results
```
eval_loss = 0.6422599390393398
eval_accuracy_score = 0.8077403245942572
eval_precision = 0.7905175085218469
eval_recall = 0.7807590287696389
eval_f1 = 0.7856079659190063
```

#### Evaluation of DEPREL and RELPOS together
Evaluation done through the script ```compute_conllu_metrics.py```
```
las = 76.61165048543688 %  (7891 / 10300)
uas = 80.85436893203884 %  (8328 / 10300)
label_acc = 92.07766990291262 %  (9484 / 10300)
total = 10300
punct = 1162
```
</details>


### Tasks Results
#### Bert_model: Musixmatch/umberto-wikipedia-uncased-v1

<details>
  <summary>Results with umberto-wikipedia-uncased-v1 model</summary>
  
#### task_type: DEPREL (Universal Dependency Relations)

##### Eval results 
```
eval_loss = 0.3033303979416968
eval_accuracy_score = 0.9387806516627477
eval_precision = 0.9369160881117682
eval_recall = 0.9338744126441691
eval_f1 = 0.9353927776826972
epoch = 3.0
```

##### Test results
```
eval_loss = 0.26721003541692356
eval_accuracy_score = 0.9451857540558702
eval_precision = 0.9435586422772181
eval_recall = 0.9412568306010929
eval_f1 = 0.942406330907137
```

#### task_type: RELPOS (Relative Position between head and id)

##### Eval results 
```
eval_loss = 0.7107215431374563
eval_accuracy_score = 0.824319785018475
eval_precision = 0.8124152886961236
eval_recall = 0.8057173581862174
eval_f1 = 0.8090524610816161
epoch = 3.0
```

##### Test results
```
eval_loss = 0.6347696009229441
eval_accuracy_score = 0.8358452529519056
eval_precision = 0.8251546391752578
eval_recall = 0.8162349581888639
eval_f1 = 0.8206705629037219
```

#### Evaluation of DEPREL and RELPOS together
Evaluation done through the script ```compute_conllu_metrics.py```
```
las = 80.45502543918595 %
uas = 83.58452529519056 %
label_acc = 94.51857540558703 %
total = 10417
punct = 1175
```
</details>
