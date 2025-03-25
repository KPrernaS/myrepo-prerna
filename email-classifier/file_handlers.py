import docx
from pdfminer.high_level import extract_text
from email import message_from_bytes
from email.policy import default
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def extract_text_from_docx(file):
        try:
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error("Failed to extract text from DOCX: %s", e)
            raise

    @staticmethod
    def extract_text_from_pdf(file):
        try:
            return extract_text(file)
        except Exception as e:
            logger.error("Failed to extract text from PDF: %s", e)
            raise

    @staticmethod
    def extract_text_from_eml(file):
        try:
            eml_content = file.read()
            msg = message_from_bytes(eml_content, policy=default)
            
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return text
        except Exception as e:
            logger.error("Failed to extract text from EML: %s", e)
            raise

    @staticmethod
    def extract_text_from_text(file):
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            logger.error("Failed to extract text from plain text file: %s", e)
            raise