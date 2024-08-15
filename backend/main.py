from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv("datasets/imdb_Dataset.csv")
    
    # Clean the text data
    df['cleaned_text'] = df['review'].apply(clean_text)
    
    # Vectorize the text data
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    return X, y, tfidf

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'\W', ' ', text)   # Remove non-word characters
    text = text.lower()
    return text

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    return model

# Load, preprocess data, and train the model
X, y, tfidf = load_and_preprocess_data()
model = train_model(X, y)

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
