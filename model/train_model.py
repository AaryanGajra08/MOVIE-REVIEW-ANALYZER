import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATASET = "IMDB Dataset.csv"
MODEL_OUT = "imdb_sentiment_model.joblib"

def clean_text(text):
    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = pd.read_csv(DATASET)

X = df["review"].map(clean_text)
y = df["sentiment"].map({"negative": 0, "positive": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=120000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="liblinear"
    ))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["negative", "positive"]))

joblib.dump(model, MODEL_OUT)
print("Saved:", MODEL_OUT)
