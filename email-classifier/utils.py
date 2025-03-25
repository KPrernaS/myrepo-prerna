import mimetypes
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

class FileUtils:
    @staticmethod
    def get_file_type(filename):
        file_type = mimetypes.guess_type(filename)[0]
        filename = secure_filename(filename).lower()
        
        if file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "docx"
        elif file_type == "application/pdf":
            return "pdf"
        elif file_type == "text/plain":
            return "text"
        elif file_type == "message/rfc822" or filename.endswith('.eml'):
            return "eml"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")