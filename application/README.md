# Application

Streamlit web application for the Internship Scam Detection project (Group 26).

A user pastes a job or internship advertisement; the app returns a label
("Likely Legitimate" / "Likely Scam"), a confidence score, and the terms that
drove the prediction.

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit interface — input form, result panel, advisory notice |
| `preprocess.py` | Shared text cleaning, also imported by the training notebooks |
| `inference.py` | Loads artefacts, runs prediction, extracts risk indicators |

## Running

```bash
pip install -r requirements.txt
streamlit run application/app.py
```

Opens on `http://localhost:8501`. Requires the artefacts in `models/`.

## Deployed model

Logistic Regression over TF-IDF features, fraud-class F1 = 0.8506 — selected over five
alternatives including three recurrent networks (Section 5 of the project report).
Loaded once at start-up with `@st.cache_resource`, so per-request cost is preprocessing
plus one sparse dot product.

## Inference path
Two failure modes this avoids: calling `fit_transform` refits the vocabulary on a single
posting and destroys the feature space; cleaning the text anywhere other than
`preprocess.py` introduces training/serving skew.

## Risk indicators

The deployed model is linear, so the explanation is the model itself — each non-zero
TF-IDF value multiplied by its coefficient, ranked. No separate rule engine or surrogate
explainer is required; the highlighted terms are the actual basis of the decision.

## Behaviour and limits

- Inputs under ~20 tokens are rejected; the model is unreliable on fragments.
- Verdicts are hedged ("Likely Scam", never "Scam") and always shown with confidence.
- Submitted text is processed in memory and never persisted.
- The corpus skews Western and English-language, so small Sri Lankan employers are more
  likely to be false-flagged. A banner states that predictions are advisory and do not
  replace verifying the employer independently.

## Not included

The BiLSTM and GRU models are research outputs only and are not loaded by the app.
See `notebooks/` and `models/README.md`.


