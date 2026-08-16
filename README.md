# Internship Scam Detection

Detecting fraudulent internship and job advertisements with Natural Language Processing.
Six models compared on 217,241 real postings; the best is deployed as a web app.

**Group 26** · Natural Language Processing · Sri Lanka Technology Campus

---

## Overview

Employment scams target students hardest — no prior experience to benchmark against,
strong pressure to secure a placement. This project frames scam detection as binary text
classification: given a posting's title, description, requirements and company profile,
predict whether it is legitimate or fraudulent.

## Results

Fraud class, held-out test set. All values from `notebooks/`.

| Model | Type | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | ML | 0.8457 | 0.8555 | **0.8506** | 0.9886 |
| XGBoost | ML | 0.9556 | 0.7457 | 0.8377 | 0.9859 |
| Random Forest | ML | 0.9907 | 0.6185 | 0.7616 | 0.9874 |
| GRU | DL | 0.6296 | 0.8844 | 0.7356 | 0.9875 |
| LSTM | DL | 0.4835 | 0.9306 | 0.6364 | 0.9847 |
| BiLSTM | DL | 0.3642 | 0.9538 | 0.5272 | 0.9841 |

**Logistic Regression is deployed.** All six models have ROC-AUC within 0.0045 of each
other, so the F1 spread reflects threshold placement rather than discriminative ability —
the fraud signal in this corpus is predominantly lexical, which favours TF-IDF over
sequence models.

Type this code: ![Image Alt](Screenshot 2026-08-16 172211.png)

## Team

| Member | ID | Models | Branch |
|---|---|---|---|
| Ashrif | CIT-24-01-0503 | Logistic Regression, LSTM | `Ashrif` |
| Nusaik | CIT-24-01-0502 | Random Forest, GRU | `Nusaik` |
| Fesal | CIT-214-01-0074 | XGBoost, BiLSTM | `Fesal` |

Preprocessing is shared, so all six models train on identical data and the comparison
isolates the models themselves.

## Quick start

```bash
git clone https://github.com/fazal2545/Internship-Scam-Detection.git
cd Internship-Scam-Detection
pip install -r requirements.txt
streamlit run application/app.py
```

The full dataset is not tracked — download it and place it at
`dataset/raw/job_postings.csv`. See `dataset/README.md`.

## Structure
