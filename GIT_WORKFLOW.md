# Git Workflow

## Branches

| Branch | Purpose |
|---|---|
| `main` | stable, CI must pass |
| `lab7-pipeline` … `lab13-integration` | one feature branch per lab, merged by pull request |

## Typical cycle

```bash
git checkout main && git pull
git checkout -b lab12-monitoring
# ...work...
git add src/mlops_pipeline/12_*.py docs/LAB12_MONITORING.md
git commit -m "Lab 12: drift detection and retraining"
git push -u origin lab12-monitoring
# open a pull request -> Lab 8 CI runs -> merge
```

## What is committed

Code, `config.yaml`, workflows, docs, `data/raw/weatherHistory.csv` and
`data/processed/dataset_metadata.json`. Generated data, models, deployment bundles,
reports, logs and `mlruns/` are ignored (`.gitignore`).

## Rolling back a faulty commit

```bash
git log --oneline
git revert <bad-commit>        # safe on shared branches
git push
```

Record conflicts you resolved and any reverts with their commit hashes as lab evidence.
