# Dataset

Data used for the Internship Scam Detection project (Group 26).

## Source

| Property | Value |
|---|---|
| Name | Internship / Job Scam Detection Dataset |
| Origin | Derived from the Employment Scam Aegean Dataset (EMSCAD) family |
| Link | https://www.kaggle.com/datasets |
| Records | 217,241 |
| Target | `fraudulent` — 0 = legitimate, 1 = scam |
| Balance | ~95% legitimate / ~5% fraudulent |
| Task | Supervised binary text classification |

## Files

| File | Description | Tracked |
|---|---|---|
| `raw/job_postings.csv` | Original download, unmodified | No — see below |
| `processed/cleaned.csv` | Output of the shared preprocessing pipeline | No |
| `sample.csv` | 200-row stratified sample for quick inspection | Yes |
| `README.md` | This file |

The full CSV is excluded via `.gitignore` — it exceeds GitHub's file size limit. Download
it from the link above and place it at `dataset/raw/job_postings.csv`. Only `sample.csv`
is tracked, so the notebooks can be smoke-tested without the full download.

## Fields

| Group | Attributes |
|---|---|
| Free text (primary signal) | `title`, `company_profile`, `description`, `requirements`, `benefits` |
| Text (secondary) | `location`, `department`, `salary_range` |
| Binary flags | `telecommuting`, `has_company_logo`, `has_questions` |
| Categorical | `employment_type`, `required_experience`, `required_education`, `industry`, `function` |
| Label | `fraudulent` |

The five free-text fields are concatenated into a single document during preprocessing.

## Known issues

- **Severe imbalance.** Fraud is ~5% of records, so accuracy is not a usable metric —
  predicting "legitimate" for everything scores ~95%.
- **Missing values.** `department`, `salary_range`, `company_profile` and `benefits` are
  frequently blank. Imputed as empty strings, not dropped — missingness is weakly informative.
- **Noise.** Scraped text retains HTML tags, URLs, escaped entities and non-ASCII characters.
- **Duplicates.** Reposted advertisements appear multiple times and must be removed
  before splitting, or they leak across partitions and inflate scores.
- **Geographic skew.** Dominated by Western, English-language job boards. Sri Lankan and
  South Asian postings are underrepresented — the main fairness limitation of the project.

## Split

Stratified 70% train / 15% validation / 15% test, seed `42`, applied after deduplication.
The same split is used by all three members so that model comparisons are valid.

## Usage

Do not clean the data inside a notebook. Import the shared function:

```python
from application.preprocess import clean_text
```

Reimplementing the cleaning introduces training/serving skew and makes the group's
results non-comparable.

## Licence

Research and educational use only, per the original dataset terms. Contains real company
names and job advertisements — do not redistribute the raw file.
