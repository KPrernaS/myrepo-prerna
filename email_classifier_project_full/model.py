
import os
import json
import re
import logging
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from pymongo import MongoClient
from hashlib import sha256
from collections import defaultdict

# Setup logging with rolling files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client.email_classifier
db.emails.create_index("email_hash", unique=True)

# Load extraction config
with open("config/extraction_config.json") as f:
    extraction_config = json.load(f)

# Load dataset
df = pd.read_csv("data/mock_emails.csv")

# Preprocess labels
all_labels = set()
for rt, srt in zip(df["request_types"], df["sub_request_types"]):
    combined = f"{rt.strip()} | {srt.strip()}"
    all_labels.add(combined)

all_labels = sorted(list(all_labels))
mlb = MultiLabelBinarizer(classes=all_labels)
label_matrix = mlb.fit_transform([[f"{r.strip()} | {s.strip()}"] for r, s in zip(df["request_types"], df["sub_request_types"])])

# Dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(all_labels), problem_type="multi_label_classification")

# Create dataset
train_dataset = EmailDataset(df["email_text"].tolist(), label_matrix, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="no",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train model
trainer.train()
model.save_pretrained("distilbert-email-classifier")
tokenizer.save_pretrained("distilbert-email-classifier")

# Inference function
def predict_email(email_text):
    # Duplicate detection
    email_hash = sha256(email_text.encode()).hexdigest()
    if db.emails.find_one({"email_hash": email_hash}):
        logging.info("Duplicate email detected.")
        is_duplicate = True
    else:
        is_duplicate = False
        db.emails.insert_one({"email_hash": email_hash, "timestamp": datetime.utcnow()})

    # Preprocessing
    logging.info("Tokenizing email text.")
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0]

    results = []
    for idx, prob in enumerate(probs):
        if prob.item() > 0.5:
            rt_srt = mlb.classes_[idx]
            rt, srt = rt_srt.split(" | ")
            decision_words = extract_decision_words(email_text, rt, srt)
            extracted_info = extract_info(email_text, rt, srt)
            results.append({
                "request_type": rt,
                "sub_request_type": srt,
                "confidence_score": round(prob.item(), 2),
                "decision_words": decision_words,
                "extracted_info": extracted_info
            })

    priority_flag = any(r["request_type"].startswith("Money Movement") for r in results)

    return {
        "email_id": email_hash,
        "is_duplicate": is_duplicate,
        "predictions": results,
        "priority_flag": priority_flag
    }

# Extract decision words (simple keyword match, extendable)
def extract_decision_words(text, request_type, sub_request_type):
    keywords = ["payment", "principal", "interest", "fee", "increase", "decrease"]
    found = [word for word in keywords if word in text.lower()]
    return found

# Extract info based on config
def extract_info(text, request_type, sub_request_type):
    fields = extraction_config.get(request_type, {}).get(sub_request_type, [])
    info = {}
    for field in fields:
        if field == "loan_amount":
            match = re.search(r"\$([\d,]+)", text)
            if match:
                info[field] = int(match.group(1).replace(",", ""))
        elif field == "adjustment_rate":
            match = re.search(r"(\d+\.?\d*)%", text)
            if match:
                info[field] = float(match.group(1))
        elif field == "fee_amount":
            match = re.search(r"\$([\d,]+)", text)
            if match:
                info[field] = int(match.group(1).replace(",", ""))
        elif field == "due_date":
            match = re.search(r"due by (\d{1,2} [A-Za-z]+)", text)
            if match:
                info[field] = match.group(1)
        elif field == "principal":
            match = re.search(r"\$([\d,]+) principal", text.lower())
            if match:
                info[field] = int(match.group(1).replace(",", ""))
        elif field == "interest":
            match = re.search(r"\$([\d,]+) interest", text.lower())
            if match:
                info[field] = int(match.group(1).replace(",", ""))
    return info

# Example usage
if __name__ == "__main__":
    email = "Please process the ongoing fee payment of $25,000 due by 31 March."
    result = predict_email(email)
    print(json.dumps(result, indent=2))
