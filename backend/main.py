from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import joblib

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React app URL
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

class TextInput(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'\W', ' ', text)   # Remove non-word characters
    text = text.lower()
    return text

@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
    cleaned_text = clean_text(input.text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    confidence = model.predict_proba(vectorized_text).max() * 100
    
    result = {
        "sentiment": prediction,
        "confidence": round(confidence, 2)
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
