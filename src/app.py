from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import predict_sentiment
import os
import psycopg2
from datetime import datetime

app = FastAPI(title="BERT Sentiment Pipeline API")

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: SentimentRequest):
    try:
        # Run inference
        prediction = predict_sentiment([request.text])[0]
        
        # Log to DB (Fire and forget, nice to have)
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                conn = psycopg2.connect(db_url)
                cur = conn.cursor()
                # Ensure table exists (simplified for demo)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        prediction TEXT,
                        timestamp TIMESTAMP
                    );
                """)
                cur.execute(
                    "INSERT INTO predictions (text, prediction, timestamp) VALUES (%s, %s, %s)",
                    (request.text, prediction, datetime.now())
                )
                conn.commit()
                cur.close()
                conn.close()
        except Exception as e:
            print(f"DB Logging failed: {e}")

        return {"text": request.text, "sentiment": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
