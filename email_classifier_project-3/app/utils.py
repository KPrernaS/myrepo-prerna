import json

def load_emails_from_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_results_to_file(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
