# SpreadsheetBench integration

This optional package provides the SpreadsheetBench workbook harness and its
`openpyxl` dependency. The deterministic answer-cell scorer remains in the
main `rethinkskill` package.

```bash
python -m pip install ./integrations/spreadsheetbench
rethinkskill --load-harness-plugins benchmark-catalog
```

Pass the task dataset with `--dataset` and the workbook corpus with
`--asset-root`. Inputs are copied into task workspaces; reference workbooks
remain verifier-only. Generated Python is checked, executed in a bounded local
process, and restricted to the supplied workbook paths.

The package does not redistribute SpreadsheetBench data. Formal use should
record the dataset, workbooks, selected IDs, package versions, skill, provider
configuration, generated code, result ledger, and receipt.
