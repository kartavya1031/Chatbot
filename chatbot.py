import json
import faiss
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import logging

# Initialize logging
logging.basicConfig(filename="chatbot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load spaCy model for query preprocessing
nlp = spacy.load("en_core_web_sm")

# Function to clean and normalize queries
def preprocess_query(query):
    doc = nlp(query)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to scrape documentation pages
def scrape_docs(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs if para.get_text()])
        logging.info(f"Successfully scraped {url}")
        return content
    except requests.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return ""

# Scrape data from CDP websites
doc_sources = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

all_docs = []

for platform, url in doc_sources.items():
    content = scrape_docs(url)
    if content:
        all_docs.append({"platform": platform, "content": content})

try:
    with open("cdp_docs.json", "w") as f:
        json.dump(all_docs, f)
    logging.info("Documentation data saved successfully")
except IOError as e:
    logging.error(f"Error writing file: {e}")

# Load and preprocess documentation data
def load_docs():
    try:
        with open("cdp_docs.json", "r") as f:
            data = json.load(f)
        logging.info("Documentation data loaded successfully")
        return data
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading documentation data: {e}")
        return []

docs = load_docs()

doc_texts = [doc["content"] for doc in docs]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(doc_texts, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
logging.info("FAISS index built successfully")

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"answer": "Please provide a valid question."}), 400

        cleaned_query = preprocess_query(user_query)
        query_embedding = model.encode([cleaned_query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=1)
        
        best_match = docs[I[0][0]] if I[0][0] >= 0 else {"content": "No relevant information found."}
        logging.info(f"Query: {user_query} - Answer: {best_match['content'][:100]}...")
        
        return jsonify({"answer": best_match["content"]})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route("/compare", methods=["POST"])
def compare_cdps():
    try:
        query = request.json.get("query", "")
        if "compare" in query.lower():
            return jsonify({"answer": "Cross-CDP comparison is under development."})
        return jsonify({"answer": "Invalid comparison request."})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Chatbot is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)
    logging.info("Chatbot service started")
