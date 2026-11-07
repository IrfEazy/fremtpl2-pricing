# Data

Source: freMTPL2freq and freMTPL2sev from the R package CASdatasets (Dutang and Charpentier),
obtained from OpenML over HTTPS. Licence as stated by OpenML: CC0. Original data:
https://github.com/dutangc/CASdatasets

`data/raw/` and `data/processed/` are git-ignored; regenerate with `make download`. The values below were
recorded from `data/raw/manifest.json` (retrieved 2026-11-07), which is the authoritative record.

| Dataset | OpenML id | Name | Version | Rows | Columns |
|---|---|---|---|---|---|
| freq | 41214 | freMTPL2freq | 1 | 678013 | 12 |
| sev | 41215 | freMTPL2sev | 1 | 26639 | 2 |

Note: the OpenML description text says 677,991 policies; the file actually contains 678,013 rows. Row counts
here are those parsed from the downloaded files.

SHA-256 of the raw files (the only hash-authoritative files, also checked against the MD5 published by OpenML):

| File | Bytes | SHA-256 |
|---|---|---|
| freMTPL2freq.arff | 36288567 | a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd |
| freMTPL2sev.arff | 376555 | 047632016b87e132247124f449a95d6d0e9f8ba05f2fe0947ef76ab8aaa10e0a |

The parquet files are derived from the ARFF by this repository (`src/fremtpl2_pricing/arff.py`). Their bytes
depend on the pyarrow/pandas versions pinned in `uv.lock`, so they are not hashed. The manifest records rows,
columns and the Arrow schema for each, and a re-run checks those; a missing or mismatching parquet is
regenerated offline from the verified ARFF. `IDpol` is int64 in both files (the parse fails if any value is
non-integral); the integer columns are listed per dataset as `int_columns` in `configs/default.yaml`.
A re-run only reuses the manifest if dataset ids, keys, names and versions match the config.

## Reproduce

```sh
make setup
make download            # fetches, writes data/raw/manifest.json; re-runs verify hashes only
uv run fremtpl2 download --force   # re-download
```
