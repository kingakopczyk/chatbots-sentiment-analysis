# From Distance to Trust

**Analysis of User Opinions on Interactions with AI Chatbots**

An NLP-driven study of how sentiment toward AI chatbots differs across use-case categories, evolves over time, and relates (or fails to relate) to star ratings — based on ~140,000 Reddit posts, comments, and app store reviews.

---

## Overview

Some people talk to chatbots every day — to save time finding information, process difficult emotions, or organise their schedule. Yet how they feel about these conversations can vary enormously, depending on how someone — or something — responds on the other side.

This project explores the emotional landscape of user opinions on human–chatbot interactions across four distinct contexts: **general-purpose AI, therapeutic, educational, and virtual assistants**. Using three complementary sentiment models, statistical hypothesis testing, topic modelling, and machine learning classification, it maps the affective content of user experience — from distance and distrust to relief and genuine connection.

## Research Questions

1. Does affect in user opinions vary depending on the chatbot category?
2. What topic-affect patterns recur across opinions?
3. Has sentiment toward chatbots changed over time?
4. Can a review's star rating be predicted from the text of the comment?

---

## Key Findings

### 1 · Affect varies significantly by chatbot category

Kruskal-Wallis tests confirm category differences in valence across all three sentiment models (p < .001), with small-to-moderate but **consistent** effect sizes:

| Model | H-statistic | ε² (valence) | ε² (arousal) |
|---|---|---|---|
| NRC VAD | 2767 | 0.020 | 0.002 *(negligible)* |
| VADER | 3467 | 0.025 | — |
| GoEmotions | 4838 | 0.035 | 0.011 *(small)* |

A one-way MANOVA on valence and arousal jointly confirms the separation is statistically robust (Wilks' λ = 0.979, F(6, 273942) = 491.21, p < .001). Categories differ **mainly in valence**, not arousal.

Dunn's post-hoc test (Bonferroni-corrected) is consistent across all three models:
- **Largest valence gap:** Educational ↔ Therapeutic
- **Smallest valence gap:** Therapeutic ↔ Virtual Assistant

### 2 · Topic-affect patterns are category-specific

BERTopic clustering mapped onto Russell's circumplex model reveals thematically distinct patterns per category — therapeutic users write about relationships and emotional support, educational users about correct/incorrect answers and paywalls, virtual assistant users about smart-home routines and device reliability, and general chatbot users about capabilities, safety, and censorship debates.

### 3 · Sentiment shifted over time — but not uniformly

Spearman correlation on monthly mean valence, by category:

| Category | Significant model(s) | Direction |
|---|---|---|
| General | VADER (ρ = +0.318, p = .040) | ↗ upward |
| Therapeutic | NRC VAD (ρ = +0.361, p = .017) | ↗ upward |
| Virtual Assistant | VADER (ρ = +0.308), NRC VAD (ρ = +0.328) | ↗ upward |
| **Educational** | VADER (ρ = −0.571), GoEmotions (ρ = −0.468) | ↘ **downward** |

Educational is the only category with a consistent declining sentiment trend — a candidate signal for product or market-level investigation.

### 4 · Star ratings are only partially predictable from text

| Classifier | Accuracy | F1 Macro | AUC-ROC |
|---|---|---|---|
| Gradient Boosting | 0.71 | 0.32 | 0.78 |
| Random Forest | 0.69 | 0.34 | 0.73 |
| **Logistic Regression** | 0.55 | **0.38** | 0.76 |
| *Majority Class baseline* | *0.53* | *0.14* | *0.50* |
| *Stratified Random baseline* | *0.37* | *0.20* | *0.50* |

All three classifiers clear both baselines on every metric. The best overall configuration — **Logistic Regression on all models combined** (F1 macro = 0.384, AUC-ROC = 0.756) — is also the most balanced across classes. The strongest single result is **1★ review detection** (Random Forest on GoEmotions: recall = 0.860, F1 = 0.666), suggesting sentiment features are more useful for flagging negative reviews than for reconstructing the full 5-point scale.

**Conclusion:** star ratings and text-derived sentiment capture related but distinct signals — they should be treated as complementary, not interchangeable, measures of user opinion.

### Inter-model agreement *(methodological finding)*

Because the project triangulates across VADER, NRC VAD, and GoEmotions, their mutual agreement was tested directly:

| Pair | Pearson r (valence) | Cohen's κ |
|---|---|---|
| VADER – NRC VAD | 0.473 | 0.241 |
| VADER – GoEmotions | 0.561 | 0.378 |
| NRC VAD – GoEmotions | 0.435 | 0.148 |

Agreement is fair-to-moderate — never strong. This is the empirical justification for reporting all three models throughout rather than collapsing to one: conclusions that hold across all three (like Finding 1 above) are considerably more trustworthy than any single-model result.

---

## Data

### Sources

| Source | Method | Volume | Notes |
|---|---|---|---|
| Reddit (posts + comments) | PullPush API, 14 subreddits | 106,661 | Nov 2022 – present; posts and comments scraped separately |
| Google Play | `google_play_scraper`, dual sort (newest / most relevant) | 29,909 | 16 apps, English/US store |
| Apple App Store | iTunes RSS feed, dual sort (recent / helpful) | 3,564 | Language-filtered via `langdetect` |

Raw collection totalled 194,039 records; cleaning and deduplication (Section 2) reduced this to **140,134 final records**.

### Final dataset composition

**By category**

| Category | Records |
|---|---|
| General | 97,855 |
| Therapeutic | 17,113 |
| Virtual Assistant | 14,625 |
| Educational | 10,541 |

**By application** (top entries)

| App | Category | Records |
|---|---|---|
| ChatGPT | General | 76,500 |
| Replika | Therapeutic | 15,373 |
| Claude | General | 11,314 |
| Amazon Alexa | Virtual Assistant | 4,194 |
| Question.AI | Educational | 3,430 |
| Nerd AI | Educational | 3,414 |
| Microsoft Copilot | Virtual Assistant | 3,351 |
| StudyX | Educational | 3,158 |
| Grok | General | 2,720 |
| Google Gemini | General | 2,685 |

*16 applications in total across 4 categories — see notebook Section 1 for the complete list.*

---

## Methodology

| Step | Stage | Description |
|---|---|---|
| 0 | Install & Import | Dependencies and library imports |
| 1 | Data Collection | Google Play, App Store, and Reddit scraping |
| 2 | Cleaning & Preprocessing | Deduplication, filtering, text normalisation (dual pipelines for lemmatised vs. raw text) |
| 3 | Exploratory Data Analysis | Distribution, temporal, and length analysis; word clouds |
| 4 | Sentiment & Arousal Analysis | VADER, NRC VAD, GoEmotions scoring |
| 5 | Inter-Model Agreement | Pearson/Spearman correlation, Cohen's κ, Bland–Altman |
| 6 | Inter-Category Differences | Kruskal-Wallis, Dunn post-hoc, effect sizes |
| 7 | Timeline of Affect | Trend analysis with Spearman correlation, by category |
| 8 | Russell's Circumplex Model | Valence-arousal mapping, MANOVA validation |
| 9 | Topic-Affect Relationship Patterns | BERTopic modelling, quadrant-level keyword/n-gram extraction |
| 10 | Star Rating Prediction | Multi-model ML classification, bias analysis, baseline comparison |

## Tech Stack

- **Scraping:** `google_play_scraper`, `requests`, `langdetect`
- **Cleaning:** `clean-text`, `contractions`, `unidecode`, `nltk`
- **EDA & Visualisation:** `matplotlib`, `seaborn`, `wordcloud`
- **Sentiment & Emotion:** `vaderSentiment`, `transformers`, `torch`, NRC-VAD Lexicon v2.1
- **Statistics:** `scipy`, `scikit-learn`, `scikit_posthocs`, `statsmodels` (MANOVA)
- **Topic Modelling:** `bertopic`
- **Machine Learning:** `scikit-learn` (Logistic Regression, Random Forest, Gradient Boosting)

---

## Repository Structure

> Suggested layout — adjust to match your actual repo.

```
.
├── README.md
├── notebook/
│   └── From_Distance_to_Trust.ipynb
├── data/
│   ├── raw/              # scraped CSVs (gitignored — see note below)
│   └── processed/        # opinions_clean.csv, opinions_emotion.csv, etc.
├── presentation/
│   └── Od_dystansu_po_zaufanie.pdf
└── requirements.txt
```

**Note on data:** raw scraped reviews and Reddit content are not included in this repository, both for size reasons and because redistributing scraped platform content may conflict with the terms of service of Reddit, Google Play, and the App Store. The notebook is structured so that Section 1 (scraping) only needs to run once — from Section 2 onward, it reads from local CSV checkpoints. Consider adding `data/raw/` to `.gitignore`.

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Open the notebook and run cells sequentially. If you don't have the pre-scraped CSVs, Section 1 will regenerate them — note that this depends on three external, unauthenticated APIs (Google Play, Apple RSS, PullPush) whose availability and rate limits are outside this project's control. GoEmotions and BERTopic steps benefit from GPU acceleration (CUDA); both fall back to CPU automatically but will run considerably slower.

---

## Limitations

- **Language and platform bias.** The dataset is limited to English-language content from Reddit, Google Play, and the App Store. Users who write public reviews or post in AI-related subreddits are not representative of the general chatbot user population.
- **Keyword-based filtering.** Technical review filtering (Section 2.5) uses keyword matching without contextual disambiguation, which risks removing genuine negative reviews that happen to mention technical vocabulary.
- **Interpolated affect coordinates.** Mapping GoEmotions categories onto Russell's circumplex (Section 4.3) involves interpolation for emotions without established psycholinguistic norms; these should be read as informed estimates, not validated values.
- **Moderate inter-model agreement.** VADER, NRC VAD, and GoEmotions agree only moderately with one another (Section 5). Findings that depend on a single model should be treated cautiously — convergence across all three is the stronger signal.
- **Temporal confounds.** Sentiment trends over time (Section 7) may partly reflect shifts in dataset composition — new apps entering the market, changes in platform activity — rather than genuine change in user affect, and should not be read as causal evidence of improving chatbot quality.

## Acknowledgments

Topic cluster labels generated by BERTopic (Section 9.2) were drafted with the assistance of Claude (Anthropic) and manually reviewed and verified by the author before inclusion.

---

## Author

**Kinga Kopczyk**
Capstone project, Postgraduate Diploma in Data Science

## License

*Add a license (e.g. [MIT](https://choosealicense.com/licenses/mit/)) if you'd like others to be able to reuse this work.*
