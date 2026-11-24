# how many examples does it take for fine-tuning to outperform few-shot prompting?

code and results for our jamia paper on medical tasks and domain adaptation. we use labeled examples from one domain to prompt or fine-tune a model, then test it on both the source and target domains.

## results

- [main scores](results/04_main_experiments/main_scores.csv) has one row per model, task, method, example count, test split, and metric.
- [results guide](results/README.md) links to the plain csv files and run index for all 651 runs on each of the eurekalert and pubmed test sets.
- [results.zip](results.zip) contains the full `results/` folder, including the guide and index.

in the scores file, `score` is macro-f1 for causal classification, negated-class f1 for negation, and strict micro-f1 for the two extraction tasks. the extraction tasks also have a `lenient` score based on word overlap. `mean` and `sd_samples` summarize the training draws. `ci2_lo` and `ci2_hi` give the 95% percentile interval from 2,000 bootstrap replicates over test items and training draws. zero-shot resamples test items only.

## prediction availability

we share parsed, item-level predictions for two publicly available datasets, eurekalert press releases and pubmed causal claims. for the other datasets, we share aggregate scores. outputs for the clinical benchmarks can reproduce text from notes covered by the providers' data-use terms. bc5cdr is also publicly downloadable, and its predictions can contain verbatim spans from pubmed abstracts whose reuse terms vary.

to reproduce predictions for another dataset, first obtain the data from the [original provider](#datasets-and-access) and follow its access and use terms. then follow the [data preparation guide](DATA.md) and the [rerun instructions](#rerunning-the-experiments).

## datasets and access

| task | dataset | role | access | original source |
|---|---|---|---|---|
| causal classification | eurekalert press releases | source | public download | [yu et al. (2020)](https://github.com/junwang4/correlation-to-causation-exaggeration#eurekalert-press-releases) |
| causal classification | pubmed causal claims | target | public download | [yu et al. (2019)](https://github.com/junwang4/causal-language-use-in-science) |
| negation classification | i2b2 2010 | source | approval and data-use agreement | [harvard dbmi data portal](https://n2c2.dbmi.hms.harvard.edu/data-sets) |
| negation classification | mimic-iii v1.4 | target | credentialed access | [physionet](https://physionet.org/content/mimiciii/1.4/) |
| disease mention detection | biocreative v cdr (bc5cdr) | source | public download | [corpus download](https://sourceforge.net/projects/bioc/files/CDR_Data.zip/download) |
| disease mention detection | share/clef ehealth | target | credentialed access | [physionet, 2014 task 2](https://physionet.org/content/shareclefehealth2014task2/1.0/) |
| ade relation extraction | n2c2 2018 ade and medication extraction | source | approval and data-use agreement | [harvard dbmi data portal](https://n2c2.dbmi.hms.harvard.edu/data-sets) |
| ade relation extraction | made 1.0 | target | clinical corpus. current access route unverified | [challenge paper](https://pubmed.ncbi.nlm.nih.gov/30649735/) |

the negation experiments use the [semeval 2021 task 10 resources](https://zenodo.org/records/5132956). check the original providers' terms before sharing source text.

## rerunning the experiments

### 1. prepare the data and write a run manifest

use python 3.12 and [uv](https://docs.astral.sh/uv/) for the local scripts. follow [data preparation](DATA.md) first. an exact rerun requires prepared files that match the study's split checksums. the [task prompts](configs/tasks.yaml), [model revisions](configs/models.yaml), and [training recipes](configs/recipe.yaml) define the runs.

from the repo root, point `FTFS_DATA` at your prepared files and `FTFS_OUT` at a private output directory. the example below needs the bc5cdr training pool and dev split. it writes a manifest for three runs. the few-shot and fine-tuning runs share the same 20 examples.

```bash
umask 077
export PYTHONPATH="$PWD/src"
export FTFS_DATA="$HOME/ftfs-data"
export FTFS_OUT="$HOME/ftfs-out"

uv run python scripts/make_manifest.py --task disease --model qwen2.5-3b \
  --mode zeroshot fewshot finetune --size 20 --sample 0 --split dev
```

### 2. train and generate

run these commands from the repo root in a cuda environment with the same `PYTHONPATH`, `FTFS_DATA`, and `FTFS_OUT` settings. we used vllm 0.30.0, pytorch 2.13.0+cu130, transformers 5.17.0, peft 0.19.1, and bitsandbytes 0.50.2. request access from the model provider for gated checkpoints.

```bash
export SPEC="$FTFS_OUT/manifests/study.json"
hf download Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1
python -m ftfs.gpu.train --specs "$SPEC"
python -m ftfs.gpu.nf4_base --specs "$SPEC"
python -m ftfs.gpu.generate --specs "$SPEC" --tp 1 --base original
python -m ftfs.gpu.generate --specs "$SPEC" --tp 1 --base nf4
```

### 3. score the outputs

```bash
uv run python scripts/score_runs.py "$FTFS_OUT/manifests/study.json"
```

this writes the aggregate scores under `$FTFS_OUT/scores/`.

### full grid and intervals

run the manifest command without filters to set up the full main test grid. qwen2.5-72b uses one gpu for training and nf4 preparation, then four gpus for inference with `--tp 4`. run the encoder with `python -m ftfs.gpu.encoder --specs "$SPEC"`.

use `uv run python scripts/bootstrap_ci.py --B 2000` on completed runs for intervals and paired comparisons. use runs from the same hardware and software environment for direct comparisons. see the paper's supplement for the sensitivity analyses.

## license

the code is released under the [Apache License 2.0](LICENSE). see [dataset access](#datasets-and-access) for the original data sources.
