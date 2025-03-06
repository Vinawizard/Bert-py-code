import os
import fitz
import numpy as np
import hashlib
import torch
from transformers import BertTokenizer, BertModel
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set Tesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load BERT Model for text embeddings
bert_model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Storage for legit certificates (in-memory dictionary)
legit_certificates = {}

# Function to extract text from a PDF (with OCR fallback for image-based PDFs)
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        if not text.strip():  # If no text is extracted, use OCR
            images = [Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples) for page in doc]
            text = " ".join([pytesseract.image_to_string(img) for img in images])
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text.strip()

# Function to create a unique hash of the certificate text
def hash_certificate(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Function to generate a BERT embedding for similarity comparison
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Route to display the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to add a legitimate certificate
@app.route("/add_legit", methods=["POST"])
def add_legit():
    if "pdf" not in request.files or "wallet_address" not in request.form:
        return jsonify({"error": "PDF and wallet address required"})
    
    pdf = request.files["pdf"]
    wallet_address = request.form["wallet_address"]
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)
    
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return jsonify({"error": "Could not extract text from PDF"})
    
    cert_hash = hash_certificate(text)
    if cert_hash in legit_certificates:
        if legit_certificates[cert_hash]["issuer"] == wallet_address:
            return jsonify({"status": "Already Exists", "issuer": wallet_address, "hash": cert_hash})
        else:
            return jsonify({
                "status": "Conflict",
                "message": "This certificate already exists and is issued by a different wallet.",
                "existing_issuer": legit_certificates[cert_hash]["issuer"]
            })
    
    # Generate embedding for the new certificate
    embedding = get_bert_embedding(text)
    if embedding is None:
        return jsonify({"error": "Failed to generate text embedding"})
    
    # Check similarity with existing certificates
    similarities = {
        hash_val: np.dot(embedding, data["embedding"]) / (np.linalg.norm(embedding) * np.linalg.norm(data["embedding"]))
        for hash_val, data in legit_certificates.items()
    }
    
    if similarities:
        max_similarity = max(similarities.values()) * 100  # Convert to percentage
        closest_hash = max(similarities, key=similarities.get)
        closest_issuer = legit_certificates[closest_hash]["issuer"]
        
        # If similarity is 90% or higher and issuer is different, flag it
        if max_similarity >= 90 and closest_issuer != wallet_address:
            return jsonify({
                "status": "Similarity Detected",
                "similarity": f"{max_similarity:.2f}%",
                "existing_issuer": closest_issuer,
                "hash": cert_hash,
                "message": f"This certificate is {max_similarity:.2f}% similar to one issued by {closest_issuer}. Are you {closest_issuer}?"
            })
    
    # If no issues, add the certificate to the storage
    legit_certificates[cert_hash] = {
        "text": text,
        "embedding": embedding,
        "issuer": wallet_address
    }
    return jsonify({"status": "Added", "issuer": wallet_address, "hash": cert_hash})

# Route to verify a certificate
@app.route("/verify", methods=["POST"])
def verify():
    if "pdf" not in request.files:
        return jsonify({"error": "PDF required"})
    
    pdf = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)
    
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return jsonify({"error": "Could not extract text from PDF"})
    
    new_hash = hash_certificate(text)
    if new_hash in legit_certificates:
        return jsonify({
            "status": "Verified",
            "issuer": legit_certificates[new_hash]["issuer"],
            "hash": new_hash,
            "similarity": "100%"
        })
    
    new_embedding = get_bert_embedding(text)
    if new_embedding is None:
        return jsonify({"error": "Failed to generate text embedding"})
    
    similarities = {
        hash_val: np.dot(new_embedding, data["embedding"]) / (np.linalg.norm(new_embedding) * np.linalg.norm(data["embedding"]))
        for hash_val, data in legit_certificates.items()
    }
    
    if similarities:
        max_similarity = max(similarities.values()) * 100
        closest_hash = max(similarities, key=similarities.get)
        closest_issuer = legit_certificates[closest_hash]["issuer"]
        if max_similarity >= 90:
            return jsonify({
                "status": "Possible Fraud",
                "similarity": f"{max_similarity:.2f}%",
                "issuer": closest_issuer,
                "hash": new_hash
            })
    
    return jsonify({"status": "Unrecognized", "hash": new_hash})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)