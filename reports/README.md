# Reports

Project documentation and evaluation outputs for **Group 26 — Internship Scam Detection**.

## Contents

| File | Description | Owner |
|---|---|---|
| `XGBOOST.png` | XGBoost evaluation output — confusion matrix and classification report | Member 03 (Fesal) |
| `BILSTM.png` | BiLSTM evaluation output — confusion matrix and training curves | Member 03 (Fesal) |
| `README.md` | This file |

## Member 03 — Fesal (CIT-214-01-0074)

**Models:** XGBoost (ML) and Bidirectional LSTM (DL)
**Branch:** `Fesal`
**Notebook:** `notebooks/member03_xgboost_bilstm.ipynb`

### Results (fraud class, held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9860 | 0.9556 | 0.7457 | 0.8377 | 0.9859 |
| BiLSTM | 0.9172 | 0.3642 | 0.9538 | 0.5272 | 0.9841 |

XGBoost ranked 2nd of six models overall. BiLSTM achieved the highest recall of any
model (0.9538) but the lowest precision (0.3642); its decision threshold was left at
the default 0.5 and has not yet been tuned on the validation set.

### Reproducing these results

1. Open the notebook in Google Colab and select a GPU runtime (Runtime → Change runtime type).
2. Mount Drive and set `DATA_PATH` in the configuration cell at the top.
3. Run all cells in order. The shared cleaning function is imported from
   `application/preprocess.py` — do not reimplement it in the notebook.
4. Random seed is fixed at `42`; the split is stratified 70/15/15.
5. Figures are written to this folder as PNG at 150 dpi.

### Dependencies

Install from the repository root:

```bash
pip install -r requirements.txt
```

## Naming convention

`<MODEL>.png` in uppercase, one file per model. Regenerate rather than edit —
figures must match the notebook that produced them.

## Outstanding

- [ ] Tune the BiLSTM decision threshold on validation and regenerate `BILSTM.png`
- [ ] Add PR-AUC alongside ROC-AUC for both models
- [ ] Export raw confusion matrix counts for Appendix A of the report
