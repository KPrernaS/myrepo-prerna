import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
import re
import logging
from logging.handlers import RotatingFileHandler

# Set up file-based logging with rotation
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('logs/email_classifier.log', maxBytes=1_000_000, backupCount=3)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change this to DEBUG, WARNING, etc., as needed
logger.addHandler(log_handler)
logger.propagate = False

class EmailClassifier:
    def __init__(self, model_path='distilbert-email-classifier'):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        with open('request_types.json', 'r') as f:
            self.request_structure = json.load(f)

    def classify(self, text):
        logger.info("Starting classification.")
        cleaned_text = self.clean_email_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        confidence, prediction = torch.max(probs, dim=0)
        decision_words = self.extract_decision_words(cleaned_text)
        sub_request = self.identify_sub_request_type(cleaned_text)
        logger.info(f"Classification complete. Prediction: {prediction.item()}, Confidence: {confidence.item():.4f}, Sub-request: {sub_request}")
        return prediction.item(), confidence.item(), decision_words, sub_request

    def clean_email_text(self, text):
        logger.info("Original email text:")
        logger.info(text)

        text = re.sub(r'(?i)(best regards|sincerely|thank you)[\s\S]+', '', text)
        logger.info("After removing signature:")
        logger.info(text)

        text = re.sub(r'(?i)On .* wrote:', '', text)
        logger.info("After removing reply history:")
        logger.info(text)

        text = re.sub(r'<[^>]+>', '', text)
        logger.info("After removing HTML tags:")
        logger.info(text)

        text = re.sub(r'\s+', ' ', text).strip()
        logger.info("After normalizing whitespace:")
        logger.info(text)

        return text

    def extract_decision_words(self, text):
        important_words = ['transfer', 'payment', 'invoice', 'refund', 'fee', 'principal', 'interest']
        decision_words = [word for word in text.split() if word.lower() in important_words]
        logger.info(f"Decision words found: {decision_words}")
        return decision_words

    def identify_sub_request_type(self, text):
        text_lower = text.lower()
        for request, sub_requests in self.request_structure.items():
            for sub in sub_requests:
                if sub.lower() in text_lower:
                    logger.info(f"Sub-request type identified: {sub}")
                    return sub
        logger.info("No sub-request type identified.")
        return None
