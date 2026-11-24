# data preparation

get the datasets from the [original providers](README.md#datasets-and-access) and follow their data-use terms. keep source data, prepared records, run specs, and raw outputs outside the repo. the [causal predictions](results/README.md#main-prediction-files) contain parsed labels and scoring flags.

## prepared files

the code reads `$FTFS_DATA/<task>/<domain>/<split>.jsonl`. each row has an `id` unique within the split, the `input` text, a `label`, and a `meta` object. negation inputs mark the event with `<e>` and `</e>`.

- classification uses a label string from [the task config](configs/tasks.yaml).
- disease recognition uses a list of mention strings.
- ade extraction uses a list of `[drug, relation type, entity]` triples.
- an empty list means there are no mentions or relations.

the source has `pool`, `dev`, and `test` splits. the target has a `test` split. the [data manifest](configs/data_manifest.json) gives the counts and sha-256 digests for the study splits. the draw index and seed recreate the training draws from the ordered pool, so matching the original draws requires the same ids, split assignments, and row order. use `--allow-different-data` when building a manifest for compatible data with different checksums.

## converting study exports

`scripts/prepare_data.py` reads the sentence-level exports below. provider downloads may need conversion to this layout. for an exact rerun, use the original study exports or reconstruct files that match the digests. we used sentence-level test exports from the original study for disease and ade. the ade test splits have 27,963 n2c2 items and 10,175 made items.

| task | inputs relative to the export directory | fields |
|---|---|---|
| causal | `causal_language_use/press_releases_causal_language_use_train_1000_sent_to_class.json`, `causal_language_use/press_releases_causal_language_use_test_sent_to_class.json`, `causal_language_use/pubmed_causal_language_use_test_sent_to_class.json` | json arrays with `question` and `gold_answer` |
| negation | `negation/i2b2/dev.json`, `negation/i2b2/test_sent_to_class_name.json`, `negation/mimic/test_sent_to_class_name.json` | json arrays with `question` and `gold_answer`. the i2b2 dev file starts the context with `context:` |
| disease | `ner/cdr/{train,dev,test}.jsonl`, `ner/ehealth/test.jsonl` | `id`, `question` with an `<input>` block, `gold_answer` with one entity per line in an `<output>` block, and optional `doc_id` and `sent_idx` |
| ade | `relation_extraction/n2c2/n2c2_train.jsonl`, `relation_extraction/n2c2/test.jsonl`, `relation_extraction/made/test.jsonl` | `id`, `input`, and gold `output` in `[relation type, drug, entity]` order. the document id is the part before the final underscore |

```bash
umask 077
export PYTHONPATH="$PWD/src"
export FTFS_DATA="$HOME/ftfs-data"
uv run python scripts/prepare_data.py --raw /path/to/study-exports
```

use `--tasks` with any of `causal`, `negation`, `disease`, and `ade` to prepare a subset.
