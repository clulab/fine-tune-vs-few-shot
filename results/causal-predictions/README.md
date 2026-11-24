# causal classification test predictions

we evaluated on 1,076 eurekalert press-release sentences and 2,061 pubmed conclusion sentences. the files here contain parsed predictions and scoring flags. [dataset sources](../../README.md#datasets-and-access)

## main results

these gzip-compressed csv files cover the full main grid for both test sets. per set, there are 444 runs for the four 3–4b llms, 86 for qwen2.5-72b, and 55 for the encoder.

| method | eurekalert test | pubmed test |
|---|---|---|
| zero-shot | [csv.gz](main-results/eurekalert/zero-shot.csv.gz) | [csv.gz](main-results/pubmed/zero-shot.csv.gz) |
| few-shot | [csv.gz](main-results/eurekalert/few-shot.csv.gz) | [csv.gz](main-results/pubmed/few-shot.csv.gz) |
| qlora fine-tuning | [csv.gz](main-results/eurekalert/fine-tuning.csv.gz) | [csv.gz](main-results/pubmed/fine-tuning.csv.gz) |
| encoder baseline | [csv.gz](main-results/eurekalert/encoder.csv.gz) | [csv.gz](main-results/pubmed/encoder.csv.gz) |

## supplementary analyses

| analysis | eurekalert test | pubmed test |
|---|---|---|
| initial learning rate, qwen2.5-72b | [csv.gz](supplementary-analyses/fine-tuning-hyperparameters/initial-72b-learning-rate/eurekalert.csv.gz) | [csv.gz](supplementary-analyses/fine-tuning-hyperparameters/initial-72b-learning-rate/pubmed.csv.gz) |
| serving weights, 3–4b llms and qwen2.5-72b | [csv.gz](supplementary-analyses/serving-weights-and-hardware/serving-weights/eurekalert.csv.gz) | [csv.gz](supplementary-analyses/serving-weights-and-hardware/serving-weights/pubmed.csv.gz) |
| prompting hardware replication, qwen2.5-72b | [csv.gz](supplementary-analyses/serving-weights-and-hardware/hardware-replication/eurekalert.csv.gz) | [csv.gz](supplementary-analyses/serving-weights-and-hardware/hardware-replication/pubmed.csv.gz) |
| causal fine-tuning replication, qwen2.5-72b on stampede3 | [csv.gz](supplementary-analyses/serving-weights-and-hardware/causal-fine-tuning-replication/eurekalert.csv.gz) | [csv.gz](supplementary-analyses/serving-weights-and-hardware/causal-fine-tuning-replication/pubmed.csv.gz) |
| gemma-3 and medgemma reasoning prompts | [csv.gz](supplementary-analyses/medgemma-and-gemma-3/reasoning-prompts/eurekalert.csv.gz) | [csv.gz](supplementary-analyses/medgemma-and-gemma-3/reasoning-prompts/pubmed.csv.gz) |

for the stop-at-user-turn analysis, we cut the smaller models' outputs at the first generated user-turn marker and rescored them. this changed one pubmed row and zero eurekalert rows. use the [replacement row](supplementary-analyses/medgemma-and-gemma-3/stop-at-user-turn/pubmed.csv.gz) to update the main file by `run_id`, `system`, and `item_id`.

## run index and coverage

[runs.csv](runs.csv) has one row per run and test set, with the model, method, budget, draw, serving weights, and macro-f1. use `prediction_file` to find the predictions, then filter by `run_id` and `system`.

| paper section | runs per dataset |
|---|---:|
| main results | 585 |
| supplementary analyses with separate generations | 66 |
| **total** | **651** |

the 24 gemma-3 and medgemma reasoning runs used the same fixed 1,000-item subset of each test set. the files contain parsed final answers. their run ids give the model, protocol, budget, and draw. all other runs use the full test sets.

## prediction columns

| column | meaning |
|---|---|
| `run_id`, `system` | run and execution environment. hardware reruns can share a run id |
| `item_id` | id in the prepared test split. join on dataset and item id to look up the gold label |
| `prediction` | parsed label, empty for invalid outputs |
| `invalid` | `1` if the output did not parse as one of the four labels |
| `length_limited` | `1` if generation stopped at its token limit |

macro-f1 averages f1 over `No claim`, `Correlational`, `Conditional causal`, and `Direct causal`. invalid outputs count as misses.
