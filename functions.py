from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd

# Load tokenizer and model once
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment" ##ProsusAI/finbert
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
LABELS = ["Negative", "Neutral", "Positive"]


THEME_KEYWORDS = {
    "Oil Price Outlook": ["price", "oil", "barrel", "market"],
    "Geopolitical Risk": ["war", "conflict", "iran", "middle east", "strait"],
    "Tariffs & Trade": ["tariff", "trade", "import", "export"],
    "Cost Inflation": ["cost", "steel", "water", "power", "price increase"],
    "Service Sector Margin Pressure": ["margin", "vendor", "services", "squeezed"],
    "Regulation & Politics": ["policy", "administration", "regulation", "government"],
    "Capital Discipline / Rig Count": ["rig", "spending", "capital", "cut", "lay down"],
    "M&A Environment": ["merger", "acquisition", "deal", "divestiture"],
    "Natural Gas Sentiment": ["natural gas", "henry hub", "lng"],
    "Interest Rates / Macro Policy": ["interest rate", "fed", "economy", "macroeconomic"],
}

def get_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits, dim=1)
    label_index = torch.argmax(scores).item()
    return LABELS[label_index]

# With some manual spot checks I found use of polarity as less precise than the model.
def get_sentiment_polarity(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def get_sentiment_subjectivity(text: str) -> float:
    return TextBlob(text).sentiment.subjectivity

def detect_themes(text: str) -> list[str]:
    themes = []
    lowered = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            themes.append(theme)
    return themes or ["Uncategorized"]

def detect_segment(text: str) -> str:
    # Simple rule: match known segment mentions
    if "e&p" in text.lower() or "exploration" in text.lower():
        return "E&P"
    elif "service" in text.lower() or "vendor" in text.lower():
        return "Services"
    else:
        return "Unknown"
    
def correlation_with_se(x, y):
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    r = df['x'].corr(df['y'])
    n = len(df)
    if n < 3:
        return r, None  # too few data points
    se = (1 - r**2) / (n - 2) ** 0.5
    return r, se