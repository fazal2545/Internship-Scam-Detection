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
