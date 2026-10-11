# Fish Health Finding Classification — Assignment 2

## What the project does
Uses the Canadian Fish Health Database to explore a supervised classification task:
predicting the recorded **etiological description** from contextual fields available
in the same historical case record (species, life stage, location and date where present).

This is deliberately described as *historical finding classification*, not as diagnosing
a live fish from water-quality measurements. The source data does not contain the
temperature/pH/dissolved-oxygen feature set assumed by some aquaculture examples.

## Dataset
Fisheries and Oceans Canada, Fish Health Database:
https://open.canada.ca/data/en/dataset/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21
CSV resource:
https://api-proxy.edh-cde.dfo-mpo.gc.ca/catalogue/records/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21/attachments/Fish_Health_Database.csv
Licence: Open Government Licence — Canada (check current source page).

Download the English CSV and save it to:
`data/Fish_Health_Database.csv`

## Run on Windows in VS Code
Activate your `.venv`, then run:
```powershell
python -m pip install -r requirements.txt
python fish_disease_detection.py
```
If you already downloaded the CSV elsewhere, copy it into the `data` folder above.

## Outputs
- `results_summary.csv`: real metrics from the run
- `results_for_report.txt`: actual metrics and per-class report
- `plots/class_distribution.png`
- `plots/confusion_matrix.png`
- `plots/feature_importance.png`

## Limitations to discuss
- Historical submitted diagnostic cases may be biased and not representative of all fish.
- Some findings are much more common than others.
- Species, location and life stage may correlate with recorded findings; correlation is not causation.
- This model is educational and not a veterinary diagnostic tool.
- Do not present metrics as proof of performance on live fish or new farms.
