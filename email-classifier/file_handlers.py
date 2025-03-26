import docx
from pdfminer.high_level import extract_text
from email import message_from_bytes
from email.policy import default
import logging
import tempfile

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
        """
        Extract text from PDF file handling multiple input types:
        - Flask FileStorage
        - SpooledTemporaryFile
        - File path (str)
        - File-like object
        """
        try:
            # Handle Flask FileStorage and SpooledTemporaryFile
            if hasattr(file, 'read') and hasattr(file, 'seek'):
                # Ensure we're at the start of the file
                file.seek(0)
                try:
                    return extract_text(file)
                except Exception as first_error:
                    # Some PDF libraries need the file saved to disk
                    try:
                        with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as temp_pdf:
                            file.seek(0)
                            temp_pdf.write(file.read())
                            temp_pdf.flush()
                            return extract_text(temp_pdf.name)
                    except Exception as second_error:
                        logger.error(f"PDF extraction failed with both methods. First error: {first_error}, Second error: {second_error}")
                        raise
            # Handle file path strings
            elif isinstance(file, str):
                return extract_text(file)
            else:
                raise ValueError("Unsupported file type. Must be file-like object or path string")
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
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