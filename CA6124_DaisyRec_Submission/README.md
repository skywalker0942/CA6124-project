# CA6124 Individual Assignment: DaisyRec + FPMC

This is the trimmed, upload-ready project copy for the assignment. It contains the DaisyRec extension, all six HyperOpt trial histories and selected configurations, all six final test result files, and the two MovieLens ratings files used by the experiments. DaisyRec is based on the teaching-demo revision `88b0a32faa1144033e969882781a8b0354958cf6`.

## Environment

- Windows 10/11 and Python 3.12 (64-bit)
- Install dependencies from the `DaisyRec` directory:

```powershell
cd <submission-folder>\DaisyRec
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-assignment.txt
```

The assignment dependency file constrains NumPy to `<2` for compatibility with this DaisyRec revision. PyTorch can run on CPU; install a CUDA-enabled PyTorch build separately if GPU execution is desired.

## Run tuning and final tests

1. Open `DaisyRec/run_assignment.py` in PyCharm and set the project interpreter to `DaisyRec/.venv`.
2. Click **Run**. The runner reads `DaisyRec/assignment_config.json`, runs HyperOpt for FPMC, MF-BPR, and ItemKNN on both datasets, reads each selected configuration, and then runs final testing.
3. To change the search, trial count, epoch count, split, or candidate count, edit `DaisyRec/assignment_config.json` before running.

The equivalent terminal command, after creating the environment and installing dependencies, is:

```powershell
.\.venv\Scripts\python.exe run_assignment.py
```

Trial histories and best validation configurations are saved under `DaisyRec/tune_res/`. Final metrics are saved under `DaisyRec/res/<dataset>/5filter/tloo/`. Existing result files in this package are the saved outputs from the completed six-combination run; rerunning the workflow will update those files.

## Contents

- `DaisyRec/daisy/`: DaisyRec source, including the registered FPMC implementation and sequence-aware data utilities.
- `DaisyRec/tune_hyperopt.py`: HyperOpt runner integrated with DaisyRec readers, splitters, recommenders, and metrics.
- `DaisyRec/run_assignment.py`: one-click PyCharm entry point for all six tune-and-test jobs.
- `DaisyRec/assignment_config.json`: experiment parameters and search spaces.
- `DaisyRec/data/`: only the raw rating files required for ML-100K and ML-1M.
- `DaisyRec/tune_res/` and `DaisyRec/res/`: saved tuning and test results.
- `DaisyRec/RESULTS_SUMMARY.md`: compact result table and completion summary.

Dataset file descriptions and sources are in `DATASETS.md`. The technical report is submitted separately; add the URL of the newly created private GitHub repository to the report before submitting it, and grant the marking team access as required by the assignment.

## Upload note

This trimmed copy contains fewer than 100 files in total, and its largest individual file is below GitHub's 25 MiB browser-upload limit. You can therefore upload the project contents in one web upload batch. GitHub documents the current browser limit as 100 files per batch and 25 MiB per file: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository.
