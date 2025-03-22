from pymongo import MongoClient, ASCENDING
import hashlib

class EmailDatabase:
    def __init__(self, uri='mongodb://localhost:27017/', db_name='emailDB'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db.emails
        self.collection.create_index([('hash', ASCENDING)], unique=True)

    def _hash_email(self, email_text):
        return hashlib.sha256(email_text.encode('utf-8')).hexdigest()

    def insert_email(self, email_data):
        email_hash = self._hash_email(email_data['text'])
        email_data['hash'] = email_hash
        try:
            self.collection.insert_one(email_data)
            return True
        except Exception:
            return False  # Duplicate detected

    def get_all_emails(self):
        return list(self.collection.find({}, {'_id': 0}))
