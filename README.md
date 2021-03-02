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
export BERT_MODEL=Musixmatch/umberto-wikipedia-uncased-v1
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
--task_type DEPREL \
--data_dir . \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--load_best_model_at_end \
--evaluation_strategy epoch \
--metric_for_best_model eval_loss \
--disable_tqdm False \
--save_total_limit 1 \
--do_train \
--do_eval \
--do_predict
```

By default, *early stopping* is set (on the *dev set*) for training, using the *eval_loss* metric.

### Tasks Results
#### Bert_model: bert-base-uncased

<details>
  <summary>Results with bert-base-uncased model</summary>
  
  #### task_type: POS (PoS-tagging)

##### Dev results 
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

##### Dev results 
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

##### Dev results 
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

##### Dev results 
```
eval_loss = 0.1996578574180603
eval_accuracy_score = 0.9547363117232113
eval_precision = 0.9531958586463593
eval_recall = 0.9517300299017514
eval_f1 = 0.9524623803009576
epoch = 7.0
```

##### Test results
```
eval_loss = 0.16917972266674042
eval_accuracy_score = 0.9584333301334357
eval_precision = 0.9574780058651027
eval_recall = 0.9557962529274004
eval_f1 = 0.9566363902724876
```

#### task_type: RELPOS (Relative Position between head and id)

##### Dev results 
```
eval_loss = 0.45070308446884155
eval_accuracy_score = 0.8899059455828014
eval_precision = 0.8824270795822831
eval_recall = 0.8783941213370373
eval_f1 = 0.8804059819463781
epoch = 10.0
```

##### Test results
```
eval_loss = 0.37038251757621765
eval_accuracy_score = 0.8948833637323605
eval_precision = 0.8859307802580381
eval_recall = 0.8825869631745384
eval_f1 = 0.8842557105626245
```

#### Evaluation of DEPREL and RELPOS together
Evaluation done through the script ```compute_conllu_metrics.py``` without consider ```punctuation marks``` and ```symbols``` - **dev set**
```
Considered punct: False
Considered sym: False
las = 89.09 %
uas = 91.99 %
label_acc = 94.91 %
punct = 0
sym = 0
<unk> = 12
```

Evaluation done through the script ```compute_conllu_metrics.py``` without consider ```punctuation marks``` and ```symbols``` - **test set**
```
Considered punct: False
Considered sym: False
las = 89.31 %
uas = 92.05 %
label_acc = 95.36 %
punct = 0
sym = 0
<unk> = 18
```
</details>


### Evaluation script
When running the evaluation script ```compute_conllu_metrics.py``` you can specify whether to consider ```punctuation marks``` and ```symbols```.

Example of use:
    ```python compute_conllu_metrics_parsing.py results/ --task_type DEPREL RELPOS --consider_punct --consider_sym --details```

### Results of 10 training instances
|     |    Dev set     |    Test set     |
|-----|----------------|-----------------|
| LAS |  88.79% ± 0.31 |  89.17% ± 0.25  |
| UAS |  91.66% ± 0.31 |  91.91% ± 0.29  |
| LA  |  94.89% ± 0.10 |  95.28% ± 0.13  |