from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import torch
import numpy as np

app = Flask(__name__)

# Load model 
MODEL_PATH = "./trained_model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Label mapping
id2label = {0: "Negative", 1: "Positive"}

# Embedding extractor
model.config.output_hidden_states = True

def get_embeddings(texts, batch_size=16):
    model.eval()
    all_embeddings = []
    dataloader = DataLoader(texts, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            outputs = model(**inputs)
            emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(emb)
    return np.vstack(all_embeddings)

# Load dataset for similiart search
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_texts = list(ds["train"]["text"])
train_embeddings = get_embeddings(train_texts)

nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
nn.fit(train_embeddings)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # Sentiment prediction
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    label = id2label[pred]

    # Similarity search
    query_emb = get_embeddings([text])
    D, I = nn.kneighbors(query_emb, n_neighbors=5)
    results = [(train_texts[i], float(D[0][j])) for j, i in enumerate(I[0])]

    return render_template("index.html", text=text, prediction=label, results=results)

# Json API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    label = id2label[pred]

    query_emb = get_embeddings([text])
    D, I = nn.kneighbors(query_emb, n_neighbors=5)
    results = [{"text": train_texts[i], "distance": float(D[0][j])} for j, i in enumerate(I[0])]

    return jsonify({"text": text, "prediction": label, "similar": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
