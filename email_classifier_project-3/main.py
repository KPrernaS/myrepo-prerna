from app.processor import EmailProcessor
from app.utils import load_emails_from_file, save_results_to_file

if __name__ == '__main__':
    processor = EmailProcessor()
    emails = load_emails_from_file('emails.txt')  # Input file with raw emails
    results = processor.process_batch(emails)
    save_results_to_file(results, 'classified_emails.json')  # Output results
    print("Processing complete. Results saved to 'classified_emails.json'.")
