from pathlib import Path
import re
import csv
import io
import json
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "imdb_sentiment_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Movie Review Analysis API",
    description="IMDb-trained NLP sentiment analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: str

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def find_aspects(text: str):
    s = text.lower()
    groups = {
        "Acting": ["acting", "actor", "actress", "cast", "performance"],
        "Story": ["story", "plot", "screenplay", "ending", "script"],
        "Direction": ["director", "direction", "directed"],
        "Music": ["music", "soundtrack", "score", "song"],
        "Visuals": ["visual", "cinematography", "camera", "effects", "cgi"],
    }
    found = []
    for aspect, words in groups.items():
        if any(word in s for word in words):
            found.append(aspect)
    return found or ["Story"]

def predict_review(review: str):
    cleaned = clean_text(review)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Review cannot be empty.")

    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities) * 100)
    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "aspects": find_aspects(cleaned),
        "review": review
    }

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(request: ReviewRequest):
    return predict_review(request.review)

def extract_reviews_from_csv(raw: str):
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames:
        normalized = {name.strip().lower(): name for name in reader.fieldnames if name}
        for key in ["review", "reviews", "text", "comment"]:
            if key in normalized:
                col = normalized[key]
                return [row.get(col, "").strip() for row in reader if row.get(col, "").strip()]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[1:] if len(lines) > 1 else lines

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    raw_bytes = await file.read()
    raw = raw_bytes.decode("utf-8-sig", errors="replace")

    if filename.endswith(".csv"):
        reviews = extract_reviews_from_csv(raw)
    elif filename.endswith(".txt"):
        reviews = [x.strip() for x in raw.splitlines() if x.strip()]
    elif filename.endswith(".json"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON must contain an array of reviews.")
        reviews = []
        for item in data:
            if isinstance(item, str):
                reviews.append(item)
            elif isinstance(item, dict):
                for key in ["review", "Review", "text", "Text", "comment"]:
                    if item.get(key):
                        reviews.append(str(item[key]))
                        break
    else:
        raise HTTPException(status_code=400, detail="Use CSV, TXT, or JSON.")

    reviews = reviews[:1000]
    results = [predict_review(review) for review in reviews]

    return {
        "filename": file.filename,
        "count": len(results),
        "results": results
    }
