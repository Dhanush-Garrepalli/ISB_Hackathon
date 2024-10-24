#Imporing required libraries
#!pip install fastapi uvicorn nest-asyncio pyngrok transformers torch scikit-learn pandas joblib gdown requests
#!pip install xgboost

################################################
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# Setting up ngrok authtoken
authtoken = "2iEKbwB5OZG2aOF2zLZNpgtThvW_6Ng7o9NAmLGCHSwLmPkBa"  # Replace with your actual authtoken
ngrok.set_auth_token(authtoken)

# Open a ngrok tunnel to the default FastAPI port 8000, specifying the port within the 'addr' field
tunnel = ngrok.connect(addr="8000")  # Adjust the port if necessary
public_url = tunnel.public_url
print(f"Public URL: {public_url}")


import re
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import joblib
import requests
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import gdown

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Allow nest_asyncio to work in Jupyter environments
nest_asyncio.apply()

# Define data preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    return text.lower()  # Convert to lowercase

# Function to download file from a public URL
def download_file(url, destination):
    if not os.path.exists(destination) or os.path.getsize(destination) == 0:
        gdown.download(url, destination, quiet=False)
        if not os.path.exists(destination) or os.path.getsize(destination) == 0:
            logging.error(f"Failed to download {destination}.")
            raise HTTPException(status_code=500, detail=f"Failed to download {destination}")

# URLs and path setup
model_files = {
    "config.json": "https://drive.google.com/uc?export=download&id=1s9Ag8YFisAtcEMc9hXSTw15wLMlcnz6R",
    "merges.txt": "https://drive.google.com/uc?export=download&id=14ETjCKd5rFailuwbxS85B-7BW28BJgYI",
    "special_tokens_map.json": "https://drive.google.com/uc?export=download&id=1HZxHafbwhV4fd9p6VgE0jBsrNy0h0Nsj",
    "tokenizer_config.json": "https://drive.google.com/uc?export=download&id=19cF9V6xGlcRy4UIgUhgZsWL67JFgE9Rm",
    "vocab.json": "https://drive.google.com/uc?export=download&id=16XBmWUhoAWGvmX6xYwiRpNZf1REAGSit"
}
model_dir = './save_models'
os.makedirs(model_dir, exist_ok=True)

# Download and verify model files
for filename, url in model_files.items():
    filepath = os.path.join(model_dir, filename)
    download_file(url, filepath)

# Load the AI detection model and tokenizer
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    ai_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
except Exception as e:
    logging.error(f"Failed to load model from Hugging Face: {e}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ai_model.to(device)

# Load the toxicity prediction model and vectorizer from GitHub
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    return joblib.load(BytesIO(response.content))

model_url = 'https://github.com/Divya-coder-isb/F-B/blob/main/best_xgboost_model.joblib?raw=true'
vectorizer_url = 'https://github.com/Divya-coder-isb/F-B/blob/main/tfidf_vectorizer.joblib?raw=true'
toxicity_model = load_model(model_url)
vectorizer = load_model(vectorizer_url)

# Function to predict AI or Human generated text
def predict_ai(text):
    ai_model.eval()
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = ai_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        ai_score = probabilities[0][1].item()
    return float(ai_score)  # Convert to standard float

# Function to predict toxicity
def predict_toxicity(text, threshold):
    transformed_input = vectorizer.transform([text])
    proba = toxicity_model.predict_proba(transformed_input)[0, 1]
    prediction = (proba >= threshold).astype(float)
    return float(proba), float(prediction)  # Convert to standard float

# Defining input data model for FastAPI
class ContentInput(BaseModel):
    text: str
    threshold: float = 0.5
    ai_score_threshold: float = 0.5

# Configuring CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI endpoint
@app.post("/verify_and_check_bias")
async def verify_and_check_bias(input: ContentInput):
    try:
        ai_score = predict_ai(input.text)
        result = {
            "classification": "Human Generated Text"
        }
        if ai_score > input.ai_score_threshold:
            proba, prediction = predict_toxicity(input.text, input.threshold)
            result = {
                "classification": "AI Generated Text",
                "probability_of_toxicity": proba,
                "prediction": "Toxic" if prediction else "Not Toxic"
            }
        return result
    except Exception as e:
        logging.error(f"Error in verify_and_check_bias: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Creating a public URL using ngrok
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

# Running the Uvicorn server
uvicorn.run(app, host='0.0.0.0', port=8000)
