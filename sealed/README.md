Sealed test slice, built once by `scripts/build_dataset.py`.

Read it only through `ksearch.eval.sealed.read_sealed(reason)`, which appends
to the committed access log `data/manifests/sealed_access.jsonl`. The only
script that does so is `scripts/confirm_sealed.py` (dry run unless `--confirm`).
Scripts that need to know which events exist use
`data/build/<build>/events_index.parquet`, which has no labels.
