from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#origin

origin = ["http://localhost:3000"] # React app

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
    # Placeholder for sentiment analysis logic
    result = {"sentiment": "Positive", "confidence": 85}  # Mock data
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)