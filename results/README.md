# results

start with [main scores](04_main_experiments/main_scores.csv) for the aggregate results across all four tasks. use [index.csv](index.csv) to find item-level causal predictions for the eurekalert and pubmed test sets. the two test sets have 1,076 and 2,061 items, respectively. [dataset sources](https://github.com/clulab/fine-tune-vs-few-shot/blob/main/README.md#datasets-and-access)

## layout

the folder numbers follow the experiment phases in the generation archive. the scores, run index, and predictions are plain csv files. `results.zip` extracts to this same `results/` directory.

| folder | what it contains | causal runs per dataset |
|---|---|---:|
| [03_hyperparameter_selection](03_hyperparameter_selection/) | test runs from the initial qwen2.5-72b fine-tuning grid at learning rate 2e-4 | 15 |
| [04_main_experiments](04_main_experiments/) | aggregate scores for all tasks, plus causal predictions for the small llms, qwen2.5-72b, and the encoder | 585 |
| [05_sensitivity_checks](05_sensitivity_checks/) | serving-weight checks and cross-cluster replications | 27 |
| [06_medgemma_reasoning_probe](06_medgemma_reasoning_probe/) | parsed final answers from the gemma-3 and medgemma reasoning runs, with a separate stop-at-user-turn rescoring file | 24 |
| **total** | | **651** |

the reasoning runs used the same fixed subset of 1,000 items per dataset. all other indexed runs used the full test sets. the stop-at-user-turn file is a replacement row for an existing run.

## main prediction files

each file contains one model and one test dataset. the llm files include zero-shot, few-shot, and fine-tuning. use the index to select a method, example count, or draw.

| model | eurekalert | pubmed |
|---|---|---|
| llama-3.2-3b | [csv](04_main_experiments/causal/small_llms/llama-3.2-3b/eurekalert.csv) | [csv](04_main_experiments/causal/small_llms/llama-3.2-3b/pubmed.csv) |
| qwen2.5-3b | [csv](04_main_experiments/causal/small_llms/qwen2.5-3b/eurekalert.csv) | [csv](04_main_experiments/causal/small_llms/qwen2.5-3b/pubmed.csv) |
| gemma-3-4b | [csv](04_main_experiments/causal/small_llms/gemma-3-4b/eurekalert.csv) | [csv](04_main_experiments/causal/small_llms/gemma-3-4b/pubmed.csv) |
| medgemma-1.5-4b | [csv](04_main_experiments/causal/small_llms/medgemma-1.5-4b/eurekalert.csv) | [csv](04_main_experiments/causal/small_llms/medgemma-1.5-4b/pubmed.csv) |
| qwen2.5-72b | [csv](04_main_experiments/causal/qwen2.5-72b/eurekalert.csv) | [csv](04_main_experiments/causal/qwen2.5-72b/pubmed.csv) |
| bioclinical modernbert | [csv](04_main_experiments/causal/encoder_bioclinical-modernbert/eurekalert.csv) | [csv](04_main_experiments/causal/encoder_bioclinical-modernbert/pubmed.csv) |

## supplementary prediction files

| experiment | eurekalert | pubmed |
|---|---|---|
| initial learning rate, qwen2.5-72b | [csv](03_hyperparameter_selection/qwen2.5-72b_initial_learning_rate/eurekalert.csv) | [csv](03_hyperparameter_selection/qwen2.5-72b_initial_learning_rate/pubmed.csv) |
| serving weights, 3–4b llms | [csv](05_sensitivity_checks/serving_weights_small_llms/eurekalert.csv) | [csv](05_sensitivity_checks/serving_weights_small_llms/pubmed.csv) |
| serving weights, qwen2.5-72b | [csv](05_sensitivity_checks/serving_weights_qwen2.5-72b/eurekalert.csv) | [csv](05_sensitivity_checks/serving_weights_qwen2.5-72b/pubmed.csv) |
| prompting hardware check, qwen2.5-72b on aces | [csv](05_sensitivity_checks/cluster_check_qwen2.5-72b_aces_side/eurekalert.csv) | [csv](05_sensitivity_checks/cluster_check_qwen2.5-72b_aces_side/pubmed.csv) |
| causal fine-tuning replication, qwen2.5-72b on stampede3 | [csv](05_sensitivity_checks/cluster_replication_causal_qwen2.5-72b_lr2e-4/eurekalert.csv) | [csv](05_sensitivity_checks/cluster_replication_causal_qwen2.5-72b_lr2e-4/pubmed.csv) |
| gemma-3 reasoning prompts | [csv](06_medgemma_reasoning_probe/gemma-3-4b/eurekalert.csv) | [csv](06_medgemma_reasoning_probe/gemma-3-4b/pubmed.csv) |
| medgemma reasoning prompts | [csv](06_medgemma_reasoning_probe/medgemma-1.5-4b/eurekalert.csv) | [csv](06_medgemma_reasoning_probe/medgemma-1.5-4b/pubmed.csv) |

for the stop-at-user-turn analysis, we cut outputs at the first generated user-turn marker and rescored them. this changed one pubmed row. apply the [replacement row](06_medgemma_reasoning_probe/stop_at_user_turn/pubmed.csv) to the matching main prediction by `run_id`, `system`, and `item_id`.

## run index and columns

[index.csv](index.csv) has one row per run and test dataset. it gives the phase, model, method, example count, draw, test size, and macro-f1. open `prediction_file` relative to this directory, then filter the rows by `run_id` and `system`. hardware reruns can share a run id.

| prediction column | meaning |
|---|---|
| `run_id`, `system` | run and execution environment |
| `item_id` | id in the prepared split. join on dataset and item id to look up gold labels |
| `prediction` | parsed label, empty for invalid outputs |
| `invalid` | `1` if the output did not parse as one of the four labels |
| `length_limited` | `1` if generation stopped at its token limit |

macro-f1 averages f1 over `No claim`, `Correlational`, `Conditional causal`, and `Direct causal`. invalid outputs count as misses. [prepared data format](https://github.com/clulab/fine-tune-vs-few-shot/blob/main/DATA.md#prepared-files)
