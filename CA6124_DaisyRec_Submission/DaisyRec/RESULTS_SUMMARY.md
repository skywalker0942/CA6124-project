# DaisyRec Assignment Results and Completion Status

## Experiment configuration

- DaisyRec revision from the teaching demo: `88b0a32faa1144033e969882781a8b0354958cf6`.
- Datasets: MovieLens 100K and MovieLens 1M, using DaisyRec `5filter` preprocessing.
- Models: FPMC (new model), MF with BPR, and ItemKNN.
- Split: TLOO; positive rating threshold: 4; uniform negative sampling.
- HyperOpt: 20 trials per model-dataset pair, maximizing validation NDCG@10.
- Final training: 10 epochs; candidate count: 1000; ranking metrics exported at K=1, 5, and 10.

## Hyperparameter tuning

Each row below is the best validation NDCG@10 recorded in the corresponding HyperOpt output. The full 20-trial records are in `tune_res/trials_*.csv`.

| Dataset | Model | Best validation NDCG@10 | Selected parameters |
|---|---|---:|---|
| ML-100K | FPMC | 0.084891 | factors=64, lr=0.01, reg_2=0.001, num_ng=1, batch_size=512 |
| ML-100K | MF (BPR) | 0.011232 | factors=32, lr=0.003, reg_1=0, reg_2=0.0001, num_ng=2, batch_size=512 |
| ML-100K | ItemKNN | 0.075920 | maxk=40 |
| ML-1M | FPMC | 0.152652 | factors=64, lr=0.001, reg_2=0.001, num_ng=2, batch_size=512 |
| ML-1M | MF (BPR) | 0.012839 | factors=64, lr=0.001, reg_1=0, reg_2=0.00001, num_ng=2, batch_size=256 |
| ML-1M | ItemKNN | 0.104565 | maxk=40 |

## Final test results

| Dataset | Model | Recall@10 | NDCG@10 | MRR@10 | Hit Ratio@10 | Precision@10 |
|---|---|---:|---:|---:|---:|---:|
| ML-100K | FPMC | 0.146055 | 0.078018 | 0.057669 | 0.146055 | 0.014606 |
| ML-100K | MF (BPR) | 0.021322 | 0.010338 | 0.007082 | 0.021322 | 0.002132 |
| ML-100K | ItemKNN | 0.125800 | 0.071733 | 0.055576 | 0.125800 | 0.012580 |
| ML-1M | FPMC | 0.262844 | 0.145576 | 0.110135 | 0.262844 | 0.026284 |
| ML-1M | MF (BPR) | 0.021213 | 0.009721 | 0.006337 | 0.021213 | 0.002121 |
| ML-1M | ItemKNN | 0.176003 | 0.091954 | 0.066739 | 0.176003 | 0.017600 |

## Short analysis

- FPMC has the highest Recall@10 and NDCG@10 on both datasets in these runs. Relative to ItemKNN, its NDCG@10 is about 8.8% higher on ML-100K and 58.3% higher on ML-1M.
- ItemKNN is the second-best model on both datasets. MF (BPR) is substantially lower, so the report should discuss the gap and avoid implying that these values establish a general ranking across all settings.
- ML-1M yields stronger FPMC and ItemKNN scores than ML-100K under this setup. The current results are single-run outputs; they do not include confidence intervals or repeated-seed variability.
- Treat these numbers as the current run's results only: the selected search spaces are small, each model received 20 trials, and training used 10 epochs. The candidate pool is controlled by DaisyRec's `cand_num=1000`; describe the evaluation protocol exactly as configured rather than calling it full-catalog unless verified against the assignment's expected protocol.

## Completion checklist

| Work item | Status | Evidence / remaining work |
|---|---|---|
| Integrate FPMC into the pinned DaisyRec codebase | Complete | Model implementation, registration, sequence data helpers, and DaisyRec training path are present. |
| HyperOpt tuning for all 3 models × 2 datasets | Complete | Six `best_params_*.json` files and six 20-row `trials_*.csv` files exist. |
| Final tests for all 3 models × 2 datasets | Complete | Six KPI CSV files exist under `res/<dataset>/5filter/tloo/`. |
| Reproducible run configuration and PyCharm entry point | Complete | `assignment_config.json` and `run_assignment.py`. |
| Assignment report | In progress | `../REPORT_TEMPLATE.md` still needs to be completed with method details, dataset statistics, result discussion, citations, and any required repository/access information. |
| Final validation against the assignment rubric | Remaining | Confirm the required split/candidate protocol, report length/format, submission packaging, and repository visibility from the assignment sheet. |

## Result files

- Best configurations: `tune_res/best_params_*.json`
- Trial histories: `tune_res/trials_*.csv`
- Test metrics: `res/ml-100k/5filter/tloo/` and `res/ml-1m/5filter/tloo/`
