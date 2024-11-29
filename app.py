from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import fitz  # PyMuPDF for PDF handling

app = Flask(__name__, template_folder='templates')  # Specify the templates folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'  # Use SQLite for simplicity
app.config['UPLOAD_FOLDER'] = 'temp'  # Folder for temporary PDF storage
db = SQLAlchemy(app)

# Database model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(5000))
    hs_result = db.Column(db.String(50))
    sent_result = db.Column(db.String(50))
    hs_feedback = db.Column(db.String(50))
    sent_feedback = db.Column(db.String(50))

# Create the database tables
with app.app_context():
    db.create_all()

# Load models and tokenizers
hs_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
hs_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")

sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# PDF text extraction function
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    doc.close()
    return text

# Text classification function
def classify_text(text):
    # Hate speech detection
    hs_inputs = hs_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hs_outputs = hs_model(**hs_inputs)
    hs_logits = hs_outputs.logits
    hs_probabilities = torch.nn.functional.softmax(hs_logits, dim=-1)

    hs_prediction = torch.argmax(hs_logits, dim=1).item()
    hs_result = "Hate Speech" if hs_prediction == 1 else "Not Hate Speech"
    hs_probability = round(hs_probabilities[0][hs_prediction].item() * 100, 2)

    # Sentiment analysis
    sent_inputs = sent_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        sent_outputs = sent_model(**sent_inputs)
    sent_logits = sent_outputs.logits
    sent_probabilities = torch.nn.functional.softmax(sent_logits, dim=-1)

    sent_label = torch.argmax(sent_probabilities, dim=1).item()
    sent_classes = ["negative", "neutral", "positive"]
    sent_result = sent_classes[sent_label]
    sent_probability = round(sent_probabilities[0][sent_label].item() * 100, 2)

    return hs_result, hs_probability, sent_result, sent_probability

temp_dir="temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""

    if request.method == "POST":
        # Check for PDF file first
        if 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
            pdf_file = request.files['pdf_file']
            temp_pdf_path = os.path.join("temp", pdf_file.filename)
            pdf_file.save(temp_pdf_path)
            
            # Extract text from the PDF
            text = extract_text_from_pdf(temp_pdf_path)
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
        elif 'text' in request.form and request.form['text'].strip() != "":
            text = request.form['text']
        else:
            # Return an error if neither text nor PDF is provided
            return "Please provide either text or a PDF file.", 400

        # Process the extracted or entered text
        hs_result, hs_prob, sent_result, sent_prob = classify_text(text)
        result = {
            "hs_result": hs_result,
            "hs_prob": hs_prob,
            "sent_result": sent_result,
            "sent_prob": sent_prob
        }

    return render_template("index.html", result=result, text=text)

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.json
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Run hate speech and sentiment analysis
    hs_result, hs_prob, sent_result, sent_prob = classify_text(text)
    return jsonify({
        "hs_result": hs_result,
        "hs_prob": hs_prob,
        "sent_result": sent_result,
        "sent_prob": sent_prob
    })

@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf_file']
    
    # Save the PDF temporarily and extract text
    temp_pdf_path = os.path.join("temp", pdf_file.filename)
    pdf_file.save(temp_pdf_path)
    
    text = extract_text_from_pdf(temp_pdf_path)
    os.remove(temp_pdf_path)
    
    # Run hate speech and sentiment analysis
    hs_result, hs_prob, sent_result, sent_prob = classify_text(text)
    return jsonify({
        "hs_result": hs_result,
        "hs_prob": hs_prob,
        "sent_result": sent_result,
        "sent_prob": sent_prob
    })

@app.route("/analyze_webpage_text", methods=["POST"])
def analyze_webpage_text():
    data = request.json
    webpage_text = data.get("webpage_text")
    
    if not webpage_text:
        return jsonify({"error": "No webpage text provided"}), 400
    
    # Run hate speech and sentiment analysis
    hs_result, hs_prob, sent_result, sent_prob = classify_text(webpage_text)
    return jsonify({
        "hs_result": hs_result,
        "hs_prob": hs_prob,
        "sent_result": sent_result,
        "sent_prob": sent_prob
    })

@app.route("/analyze_selected_text", methods=["POST"])
def analyze_selected_text():
    data = request.json
    selected_text = data.get("selected_text")
    
    if not selected_text:
        return jsonify({"error": "No selected text provided"}), 400
    
    # Run hate speech and sentiment analysis
    hs_result, hs_prob, sent_result, sent_prob = classify_text(selected_text)
    return jsonify({
        "hs_result": hs_result,
        "hs_prob": hs_prob,
        "sent_result": sent_result,
        "sent_prob": sent_prob
    })



# Retraining counters for each model
hs_feedback_counter = 0
sent_feedback_counter = 0

@app.route('/', methods=['GET', 'POST'])
def home():
    retrain_message = None  # Initialize the message variable
    if request.method == 'POST':
        input_text = request.form['text']
        hs_result, hs_prob, sent_result, sent_prob = classify_text(input_text)
        return render_template('index.html', text=input_text, 
                               hs_result=hs_result, hs_prob=hs_prob, 
                               sent_result=sent_result, sent_prob=sent_prob,
                               retrain_message=retrain_message)  # Pass the message

    return render_template('index.html', retrain_message=retrain_message)  # Pass the message

@app.route('/feedback', methods=['POST'])
def feedback():
    global hs_feedback_counter, sent_feedback_counter
    
    retrain_message = None

    feedback_data = Feedback(
        text=request.form.get('text', ''),
        hs_result=request.form.get('hs_result', ''),
        sent_result=request.form.get('sent_result', ''),
        hs_feedback=request.form.get('hs_feedback', ''),
        sent_feedback=request.form.get('sent_feedback', '')
    )

    # Add feedback data to the database
    db.session.add(feedback_data)
    db.session.commit()

    # Update feedback counters based on feedback type
    if feedback_data.hs_feedback:
        hs_feedback_counter += 1
    if feedback_data.sent_feedback:
        sent_feedback_counter += 1

    # Check if models need retraining
    if hs_feedback_counter >= 16:
        retrain_message = retrain_hs_model()
        hs_feedback_counter = 0
    if sent_feedback_counter >= 16:
        retrain_message = retrain_sent_model()
        sent_feedback_counter = 0

    # Render the same template with feedback message
    return render_template('index.html', message='Feedback received successfully!', retrain_message=retrain_message)


def retrain_hs_model():
    # Retrieve feedback for hate speech model
    feedback_entries = Feedback.query.all()
    
    # Prepare data for training
    data = []
    for entry in feedback_entries:
        # Map feedback to the training data format
        if entry.hs_feedback == 'correct':
            data.append((entry.text, 1 if entry.hs_result == "Hate Speech" else 0))  # Assuming 1 for hate speech
        elif entry.hs_feedback == 'false_positive':
            data.append((entry.text, 0))  # False positive is not hate speech
        elif entry.hs_feedback == 'false_negative':
            data.append((entry.text, 1))  # False negative is hate speech
            
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['text', 'label'])

    
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1)

    # Dataset class
    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            return inputs, label

    # Create datasets
    train_dataset = FeedbackDataset(train_df['text'].tolist(), train_df['label'].tolist(), hs_tokenizer)
    val_dataset = FeedbackDataset(val_df['text'].tolist(), val_df['label'].tolist(), hs_tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Optimizer
    optimizer = AdamW(hs_model.parameters(), lr=5e-5)

    # Training loop
    hs_model.train()
    for epoch in range(3):  # Adjust epochs as needed
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = hs_model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Save the retrained model
    hs_model.save_pretrained("cardiffnlp/twitter-roberta-base-hate-latest-retrained")
    return "Hate Speech Model retrained!"  # Return a message indicating retraining



def retrain_sent_model():
    # Retrieve feedback for sentiment model
    feedback_entries = Feedback.query.all()
    
    # Prepare data for training
    data = []
    for entry in feedback_entries:
        # Map feedback to the training data format
        if entry.sent_feedback == 'correct':
            data.append((entry.text, 1 if entry.sent_result == "positive" else 0))  # Assuming 1 for positive sentiment
        elif entry.sent_feedback == 'false_negative':
            data.append((entry.text, 1))  # False negative is positive sentiment
        elif entry.sent_feedback == 'false_positive':
            data.append((entry.text, 0))  # False positive is not positive sentiment
            
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1)

    # Dataset class
    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            return inputs, label

    # Create datasets
    train_dataset = FeedbackDataset(train_df['text'].tolist(), train_df['label'].tolist(), sent_tokenizer)
    val_dataset = FeedbackDataset(val_df['text'].tolist(), val_df['label'].tolist(), sent_tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Optimizer
    optimizer = AdamW(sent_model.parameters(), lr=5e-5)

    # Training loop
    sent_model.train()
    for epoch in range(3):  # Adjust epochs as needed
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = sent_model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Save the retrained model
    sent_model.save_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest-retrained")

    return "Sentiment Analysis Model retrained!"  # Return a message indicating retraining

if __name__ == '__main__':
    app.run(debug=True, port=5012)
    CORS(app)
