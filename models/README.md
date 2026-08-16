# Member 03 — Fesal (CIT-214-01-0074)

**Models:** XGBoost (ML) and Bidirectional LSTM (DL)
**Branch:** `Fesal`
**Notebook:** `notebooks/member03_xgboost_bilstm.ipynb`

## Artefacts

| File | Description | Tracked |
|---|---|---|
| `xgboost_model.joblib` | Trained XGBoost classifier | Yes |
| `xgboost_tfidf.joblib` | TF-IDF vectoriser fitted for XGBoost (10,000 features) | Yes |
| `bilstm_model.keras` | Trained BiLSTM checkpoint | No — see below |
| `bilstm_tokenizer.pkl` | Keras tokeniser, 20,000-word vocabulary | Yes |
| `bilstm_thresholds.json` | Decision thresholds for both models | Yes |

The BiLSTM checkpoint is excluded via `.gitignore` — the 20,000 × 100 embedding matrix
pushes the file well past GitHub's size limit. Regenerate it by running the notebook.
The tokeniser is small and is tracked, since inference cannot rebuild it.

## Results (fraud class, held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9860 | 0.9556 | 0.7457 | 0.8377 | 0.9859 |
| BiLSTM | 0.9172 | 0.3642 | 0.9538 | 0.5272 | 0.9841 |

XGBoost placed 2nd of the six group models. The BiLSTM achieved the highest recall of
any model but the lowest precision; its threshold was left at the default 0.5 and has
not been tuned. Neither model was selected for deployment — Logistic Regression
(Member 01) took the highest F1 at 0.8506.

## Configuration

**XGBoost:** `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`,
`scale_pos_weight≈19`, `tree_method="hist"`, early stopping on validation AUC-PR.

**BiLSTM:** Embedding 20,000 × 100 (GloVe 6B.100d, trainable) → SpatialDropout1D 0.3 →
Bidirectional LSTM 128 per direction, `return_sequences=True` → GlobalMaxPool +
GlobalAvgPool concatenated (512) → Dense 64 ReLU → Dropout 0.5 → Dense 1 sigmoid.
Adam 1e-3, `clipnorm=1.0`, class-weighted binary cross-entropy, batch 64, early stopping
on validation F1 with patience 3.

## Loading

```python
import joblib, json, pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from application.preprocess import clean_text

# XGBoost
xgb  = joblib.load("models/xgboost_model.joblib")
tfv  = joblib.load("models/xgboost_tfidf.joblib")
prob = xgb.predict_proba(tfv.transform([clean_text(text)]))[0, 1]

# BiLSTM
bilstm = load_model("models/bilstm_model.keras")
tok    = pickle.load(open("models/bilstm_tokenizer.pkl", "rb"))
seq    = pad_sequences(tok.texts_to_sequences([clean_text(text)]),
                       maxlen=300, padding="post", truncating="post")
prob   = float(bilstm.predict(seq)[0, 0])
```

Always clean text with `application/preprocess.py` — reimplementing it introduces
training/serving skew. Call `transform`, never `fit_transform`.

## Reproducing

Open the notebook in Colab, select a GPU runtime, mount Drive, set `DATA_PATH` in the
config cell, and run all cells. Seed is fixed at `42`; split is stratified 70/15/15.
Checkpoints write to Drive so a disconnect does not cost a full run.

## Outstanding

- [ ] Tune the BiLSTM threshold on validation and re-export `bilstm_thresholds.json`
- [ ] Add PR-AUC for both models
- [ ] Export confusion matrix counts for Appendix A
