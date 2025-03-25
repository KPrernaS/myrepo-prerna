import openai
import json
import logging
import re

logger = logging.getLogger(__name__)

class Classifier:
    def __init__(self, api_key):
        openai.api_key = api_key

    def normalize_content(self, content):
        return re.sub(r"\s+", " ", content).strip()

    def generate_prompt(self, text, request_types):
        request_type_list = []
        for request_type, sub_types in request_types.items():
            if sub_types:
                request_type_list.append(f"{request_type}: {', '.join(sub_types)}")
            else:
                request_type_list.append(request_type)
        request_types_str = "\n".join([f"- {rt}" for rt in request_type_list])

        prompt = f"""Analyze the following email content and identify all individual requests within it:
        Email Content: {text}

        For each request, provide the following details:
        1. Request Type: Choose from:
        {request_types_str}
        2. Sub Request Type: Choose from sub-types under selected Request Type.
        3. Confidence Score: 0-1 confidence score.
        4. Decision Words: Key phrases that influenced decision.
        5. Required Info: Important info as key-value pairs.
        6. Duplicate Flag: True/False.
        7. Priority Flag: Low/Medium/High. Money Movement requests always High.

        Return as JSON array of request classifications."""
        return prompt

    def classify_text(self, text, request_types):
        prompt = self.generate_prompt(text, request_types)
        
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
            return []
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            return []

    def merge_duplicate_requests(self, classifications):
        merged = {}
        for classification in classifications:
            key = (classification["Request Type"], classification["Sub Request Type"])
            if key in merged:
                existing = merged[key]
                existing["Required Info"].update(classification["Required Info"])
                existing_words = set(existing["Decision Words"].split(", "))
                new_words = set(classification["Decision Words"].split(", "))
                existing["Decision Words"] = ", ".join(existing_words.union(new_words))
                if classification["Confidence Score"] > existing["Confidence Score"]:
                    existing["Confidence Score"] = classification["Confidence Score"]
                if classification["Duplicate Flag"]:
                    existing["Duplicate Flag"] = True
                if classification["Priority Flag"] == "High":
                    existing["Priority Flag"] = "High"
            else:
                merged[key] = classification
        return list(merged.values())

    def enforce_priority(self, classifications):
        for classification in classifications:
            if classification.get("Request Type") in ["Money Movement – Inbound", "Money Movement – Outbound"]:
                classification["Priority Flag"] = "High"
        return classifications