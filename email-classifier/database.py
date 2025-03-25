import pymongo
import hashlib
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string="mongodb://localhost:27017", db_name="email_classification"):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db["request_hashes"]
        logger.info("Connected to MongoDB successfully")

    def generate_hash(self, content):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_duplicate(self, content_hash):
        return self.collection.find_one({"hash": content_hash}) is not None

    def store_hash(self, content_hash):
        self.collection.insert_one({"hash": content_hash})
        logger.debug("Hash stored in MongoDB")