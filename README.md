# 🔎 TruthLens - Fake News Detection System

TruthLens is a machine learning web app that reads a piece of news text and tells you whether it looks **REAL** or **FAKE**. It also pulls in **live headlines** from well-known news RSS feeds and checks them automatically, so you can see what today's news looks like through the model's eyes - or paste in any text of your own and check it on the spot.

Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [The Model, In Detail](#the-model-in-detail)
- [News Sources](#news-sources)
- [Limitations](#limitations)
- [Ideas for Future Improvement](#ideas-for-future-improvement)
- [Author](#author)

---

## Overview

Fake news spreads fast, and it isn't always obvious at a glance. TruthLens gives you a quick, second opinion: type or paste in a headline or short article, and a trained machine learning model will estimate whether it reads like real, verified reporting or like a misleading/fabricated claim - along with a confidence score, not just a flat yes/no.

Beyond checking your own text, TruthLens also has a live dashboard that fetches current headlines from several major news outlets (Times of India, Indian Express, NDTV, BBC, and BBC Hindi) and runs every one of them through the same model automatically, so you get a running, real-time view of what's being flagged.

This is a learning/demo project, not a fact-checking authority - see [Limitations](#limitations) for an honest note on what it can and can't do.

Important: A machine-learning prediction is not the same as independent fact verification. News should be cross-checked with reliable sources.

## Key Features

**Live news dashboard**
- Pulls current headlines from multiple RSS feeds across five news outlets
- Automatically marks every headline as `FAKE` or `REAL`
- Shows a quick summary: how many headlines were checked, how many were flagged, and which ones

**Check any headline**
- Pick any live headline and see the model's prediction, confidence score, and the probability split between FAKE and REAL
- If you edit a live headline's text before checking it, the app warns you and asks you to use the original text - so the "live" check stays honest

**Check your own text**
- Paste any headline or short article into a text box and get an instant prediction

**Model transparency**
- A dedicated "Model Info" page explains exactly what algorithm is running and how to read the results
- Sidebar always shows the current model's training data size and accuracy, so you know what you're looking at

**Clean, responsive interface**
- Custom dark-themed UI built with Streamlit, with a sidebar for navigation and source selection
- Works on both desktop and mobile screen sizes

## How It Works

1. **You provide text.** Either by picking a live headline from the dashboard, or by pasting your own text into the "Check Text" tab.
2. **The text is converted into numbers.** A TF-IDF vectorizer turns the raw text into a set of weighted word and word-pair features - essentially, a numerical fingerprint of which words and phrases show up, and how distinctive they are.
3. **A trained classifier makes a prediction.** A Logistic Regression model, trained ahead of time on labeled examples, looks at that fingerprint and estimates the probability the text is FAKE versus REAL.
4. **You get a result, not just a label.** TruthLens shows the predicted class (FAKE or REAL), a confidence percentage, and the individual probability for each class - so a 51% call and a 98% call don't look the same.

The live news side works the same way, just automated: the app fetches fresh RSS headlines, runs each one through the same three steps above, and displays the results in a table.

## Tech Stack

| Tool | Used for |
|---|---|
| **Python** | Core language for the whole project |
| **Streamlit** | The web app / user interface |
| **scikit-learn** | The TF-IDF vectorizer and Logistic Regression model |
| **pandas** | Loading and handling data (training data, live news, results tables) |
| **joblib** | Saving and loading the trained model |
| **urllib + xml.etree** (Python standard library) | Downloading and parsing RSS/Atom news feeds - no extra scraping library needed |

## Project Structure

```text
Fake News Detection/
├── app.py                     # The Streamlit web app (UI + prediction flow)
├── train_model.py             # Command-line script to train/retrain the model
├── scrape_news.py             # Command-line script to fetch live RSS news
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── detector.py            # ML pipeline: training, saving/loading, predictions
│   └── news_scraper.py        # RSS/Atom feed fetching and cleanup
│
├── data/
│   └── news_sample.csv        # Small sample dataset used to train the demo model
│
└── models/
    ├── fake_news_model.joblib # The trained model (created after training)
    └── metrics.json           # Accuracy and training details for the saved model
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip (comes with Python)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/truthlens-fake-news-detection.git
   cd "Fake News Detection"
   ```

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

That's it - no API keys, no external accounts, and no database setup needed. Everything runs locally.

## Usage

### Run the app

```bash
python train_model.py
python -m streamlit run app.py
```

The first command trains the model on the sample dataset (only needed the first time, or after you change the training data). The second command starts the app - Streamlit will print a local URL in your terminal; open it in your browser.

### Fetch live news from the terminal

You don't need to open the app to pull live headlines - this also works as a standalone command-line tool:

```bash
python scrape_news.py --limit-per-feed 20
```

To save the results to a CSV file instead of just printing them:

```bash
python scrape_news.py --limit-per-feed 20 --output data/latest_news.csv
```

### Retrain with your own data

Swap in your own labeled dataset and retrain:

```bash
python train_model.py --data path/to/your_dataset.csv
```

## The Model, In Detail

**Algorithm:** TF-IDF (unigrams and bigrams, English stop words removed) feeding into a Logistic Regression classifier, with balanced class weights so the model doesn't just favor whichever label happens to be more common in the training data.

**Current demo model's accuracy:** 80%, measured on a held-out validation set. Worth knowing: the current sample dataset only has 40 rows in total (20 FAKE, 20 REAL), with just 10 of those held out for validation - so this accuracy number is illustrative of how the pipeline works, not a reliable real-world performance estimate. See [Limitations](#limitations) below.

**Training data format:** any CSV with a text column and a label column works. The app is flexible about column names and label spelling:

```csv
text,label
"Example verified article text",REAL
"Example misleading article text",FAKE
```

- Text column can be named: `text`, `title`, `headline`, `article`, `content`, or `news`
- Label column can be named: `label`, `target`, `class`, or `category`
- Accepted label values: `FAKE` / `REAL`, `false` / `true`, `0` / `1`, `misleading` / `reliable`, `rumor` / `genuine` (not case-sensitive)

## News Sources

The live dashboard currently pulls from these outlets (13 RSS feeds in total):

| Source | Feeds included |
|---|---|
| Times of India | Top Stories, Most Recent |
| Indian Express | Latest, India |
| NDTV | Top Stories, India, World |
| BBC | Top Stories, World, Technology, Business, Asia |
| BBC Hindi | Latest |

You can pick and choose which sources to include right from the app's sidebar.

## Limitations

Being upfront about what this project is - and isn't:

- **The demo model is trained on a very small, mostly clear-cut dataset.** The sample data's REAL examples read like formal news reporting, and the FAKE examples are obvious, exaggerated claims. Real-world misinformation is often far more subtle, so don't expect this exact model to catch cleverly-written fake news - retraining on a larger, more realistic dataset would make a real difference here.
- **A prediction is a statistical guess, not a fact-check.** The model looks for word patterns it has seen before; it doesn't verify claims against real sources. Always cross-check anything important with trusted, reliable outlets.
- **RSS feeds can occasionally fail or go temporarily unavailable.** The app is built to keep working even if one feed is down (you'll see the error listed rather than the whole dashboard breaking), but a source may show fewer results than expected from time to time.

## Ideas for Future Improvement

- Train on a larger, more diverse, real-world labeled dataset
- Try stronger models (e.g. gradient boosting, or a fine-tuned transformer model) and compare accuracy against the current TF-IDF + Logistic Regression baseline
- Add source-credibility signals (publisher history, domain reputation) alongside the text-only prediction
- Show *why* the model made a call - e.g. highlighting the words/phrases that pushed the prediction toward FAKE or REAL
- Add more news sources and languages

## Disclaimer
TruthLens is an automated machine-learning based classification tool. It should not be treated as an authoritative fact-checking service. Always verify important claims using multiple reliable and independent sources.


## Author

**[Mohammad Somama]**
 Email: [your.email@example.com](mailto:your.email@example.com)
 [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)



