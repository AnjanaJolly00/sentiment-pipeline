from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_NAME = "anjanajolly/amazon-sentiment-distilbert"

app = FastAPI(title="Amazon Review Sentiment API")

sentiment_model = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1  # runs on CPU (safe for deployment)
)

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: list[str]

def format_prediction(prediction):
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }

    return {
        "sentiment": label_map.get(prediction["label"], prediction["label"]),
        "confidence": round(prediction["score"], 4)
    }

@app.get("/")
def home():
    return {
        "message": "Amazon Review Sentiment API is running"
    }

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    prediction = sentiment_model(request.review)[0]
    return format_prediction(prediction)

@app.post("/predict-batch")
def predict_batch_sentiment(request: BatchReviewRequest):
    batch_size = 32
    all_results = []

    for i in range(0, len(request.reviews), batch_size):
        batch = request.reviews[i:i + batch_size]
        predictions = sentiment_model(batch)

        formatted = [
            format_prediction(pred)
            for pred in predictions
        ]

        all_results.extend(formatted)

    return {"results": all_results}