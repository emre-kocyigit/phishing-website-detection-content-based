# Phishing Website Detection — Content-Based ML Pipeline

End-to-end machine learning pipeline for detecting phishing websites using 
HTML content-based features. Covers data collection, feature engineering, 
model training, and evaluation across five classifiers.

> Related publication: Korkmaz, M., Kocyigit, E. et al. (2022). *A hybrid 
> phishing detection system using deep learning-based URL and content analysis.* 
> Elektronika ir Elektrotechnika, 28(5).

---

## Pipeline overview
Phishing URLs (PhishTank)          Legitimate URLs (Tranco)
│                                   │
└──────────┬────────────────────────┘
▼
data_collector.py
(HTTP requests + BeautifulSoup)
│
▼
feature_extraction.py
(HTML tag-based features → numerical vectors)
│
▼
structured_data_phishing.csv
structured_data_legitimate.csv
│
▼
machine_learning.py
(train / evaluate / compare)
│
▼
Results + confusion matrices

## Features extracted

Content-based features parsed directly from HTML — no URL or domain signals:

- Presence and counts of HTML tags (`<form>`, `<iframe>`, `<script>`, `<a>`, etc.)
- External resource ratios (scripts, images, links)
- Form action attributes and submission targets
- Favicon and redirect indicators
- 30+ binary and quantitative features per page

## Models & results

Five classifiers trained with 5-fold cross-validation:

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Random Forest | ~97% | ~97% | ~97% |
| AdaBoost | ~96% | ~96% | ~96% |
| Decision Tree | ~95% | ~95% | ~95% |
| Naive Bayes | ~94% | ~93% | ~95% |
| SVM | ~93% | ~93% | ~93% |

> Naive Bayes achieves the best recall — minimising false negatives 
> (missed phishing pages) at the cost of slightly lower precision.

## Dataset

| Source | Type | Size |
|---|---|---|
| [PhishTank](https://phishtank.org) | Phishing URLs | ~10k |
| [Tranco List](https://tranco-list.eu) | Legitimate URLs | ~10k |

Pre-collected structured data included: `structured_data_phishing.csv` 
and `structured_data_legitimate.csv`. To rebuild from fresh URLs, run 
`data_collector.py` with your own URL lists.

## Quickstart

```bash
git clone https://github.com/emre-kocyigit/phishing-website-detection-content-based.git
cd phishing-website-detection-content-based
pip install -r requirements.txt
```

**Run feature extraction** (on your own URL list):
```bash
python data_collector.py
python feature_extraction.py
```

**Train and evaluate models:**
```bash
python machine_learning.py
```

**Flask app** (predict a single URL):
```bash
python app.py
# visit http://localhost:5000
```

## Project structure
├── data_collector.py        # Collects HTML content from URLs
├── feature_extraction.py    # Extracts content-based features
├── features.py              # Feature definitions
├── machine_learning.py      # Model training, K-fold CV, evaluation
├── app.py                   # Flask prediction interface
├── structured_data_*.csv    # Pre-built datasets (ready to use)
└── requirements.txt

## Tech stack

`Python` `Scikit-learn` `BeautifulSoup` `Pandas` `Flask` `Matplotlib`

---

*Part of a broader research programme on cyberthreat detection. 
See also: [content-based-feature-extraction](https://github.com/emre-kocyigit/content-based-feature-extraction)*
