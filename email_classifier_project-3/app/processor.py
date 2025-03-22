import json
from app.model import EmailClassifier
from app.db import EmailDatabase

PRIORITY_CATEGORIES = ['Money Movement – Inbound', 'Money Movement – Outbound']

class EmailProcessor:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.db = EmailDatabase()
        with open('categories.json', 'r') as f:
            self.category_map = json.load(f)

    def process_batch(self, emails):
        results = []
        sorted_emails = sorted(emails, key=self._priority_key, reverse=True)

        for email in sorted_emails:
            pred, conf, decision_words, sub_request = self.classifier.classify(email)
            category = self._map_category(pred)
            email_data = {
                'text': email,
                'category': category,
                'sub_request_type': sub_request,
                'confidence': conf,
                'decision_words': decision_words
            }
            inserted = self.db.insert_email(email_data)
            email_data['inserted'] = inserted
            results.append(email_data)

        return results

    def _priority_key(self, email):
        pred, _, _, _ = self.classifier.classify(email)
        category = self._map_category(pred)
        return 1 if category in PRIORITY_CATEGORIES else 0

    def _map_category(self, pred_idx):
        return self.category_map.get(str(pred_idx), 'Unknown')
