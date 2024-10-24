# Import required libraries
import streamlit as st
import requests
import os
import re
import joblib
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import importlib
import xgboost

# Define a function to download the file from the URL if not present locally
def download_file(url, destination):
    if not os.path.exists(destination) or os.path.getsize(destination) == 0:
        response = requests.get(url)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            f.write(response.content)

# Define preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    return text.lower()  # Convert to lowercase

# URLs and path setup
model_dir = './save_models'
os.makedirs(model_dir, exist_ok=True)

model_files = {
    "config.json": "https://drive.google.com/uc?export=download&id=1s9Ag8YFisAtcEMc9hXSTw15wLMlcnz6R",
    "merges.txt": "https://drive.google.com/uc?export=download&id=14ETjCKd5rFailuwbxS85B-7BW28BJgYI",
    "special_tokens_map.json": "https://drive.google.com/uc?export=download&id=1HZxHafbwhV4fd9p6VgE0jBsrNy0h0Nsj",
    "tokenizer_config.json": "https://drive.google.com/uc?export=download&id=19cF9V6xGlcRy4UIgUhgZsWL67JFgE9Rm",
    "vocab.json": "https://drive.google.com/uc?export=download&id=16XBmWUhoAWGvmX6xYwiRpNZf1REAGSit"
}

# Download and verify model files
for filename, url in model_files.items():
    filepath = os.path.join(model_dir, filename)
    download_file(url, filepath)

# Load AI detection model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
ai_model = RobertaForSequenceClassification.from_pretrained('roberta-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ai_model.to(device)

# Load the toxicity prediction model and vectorizer from GitHub
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()
    try:
        return joblib.load(BytesIO(response.content))
    except ModuleNotFoundError as e:
        module_name = str(e).split("No module named '")[1].split("'")[0]
        importlib.import_module(module_name)
        return joblib.load(BytesIO(response.content))

model_url = 'https://github.com/Divya-coder-isb/F-B/blob/main/best_xgboost_model.joblib?raw=true'
vectorizer_url = 'https://github.com/Divya-coder-isb/F-B/blob/main/tfidf_vectorizer.joblib?raw=true'

# Load models
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

# Streamlit app
# Load the banner image from the URL
banner_url = "https://github.com/Griffender/Fake-berry/raw/main/Banner.png"
banner_image = Image.open(requests.get(banner_url, stream=True).raw)

# Display the banner image as a header
st.image(banner_image, use_column_width=True)

st.title("Fake Berry")
st.write("Craft Your Query: Unmask Bias, Confirm Fairness and validate Authenticity")

# Creating columns for layout
left_col, right_col = st.columns(2)

# Input text box to capture user's input
input_text = left_col.text_area("Input Text", height=200)
ai_score_threshold = left_col.slider("AI Score Threshold", 0.0, 1.0, 0.5)

if left_col.button("Classify"):
    if input_text:
        # Predict if the text is AI generated
        ai_score = predict_ai(input_text)
        classification = "Human Generated Text"
        if ai_score > ai_score_threshold:
            classification = "AI Generated Text"

        left_col.write("**Classification Result:**")
        left_col.write(f"Classification: {classification}")

        # If AI generated, predict toxicity
        if classification == "AI Generated Text":
            proba, prediction = predict_toxicity(input_text, ai_score_threshold)
            prediction_text = "Toxic" if prediction else "Not Toxic"

            left_col.write(f"Prediction: {prediction_text}")

            # Plot the circular progress chart for toxicity score
            fig, ax = plt.subplots()
            ax.pie([proba, 1 - proba], 
                   startangle=90, colors=['#FF6F61', '#E0E0E0'], 
                   wedgeprops={'width': 0.3})
            ax.text(0, 0, f"{int(proba * 100)}%", 
                    ha='center', va='center', fontsize=20, color='#FF6F61')
            ax.set_aspect('equal')

            # Display the chart in the right column
            with right_col:
                st.pyplot(fig)
        else:
            left_col.error("Error: Unable to classify the text for toxicity. Please try again later.")
    else:
        left_col.warning("Please enter some text to classify.")
