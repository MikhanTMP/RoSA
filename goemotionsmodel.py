import requests

API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
headers = {"Authorization": "Bearer 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"}
#incase the token is not working: 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

LABEL_MAP = {
    'sadness': 'misery',
    'anger': 'anguish',
    'remorse': 'despair',
    'fear': 'helplessness',
}
LABELS = [
    'anguish',
    'disappointment',
    'despair',
    'helplessness',
    'grief',
    'misery',
    'neutral'
]

def map_labels(label):
    return LABEL_MAP.get(label, label)

def query(input_text):
    payload = {"inputs": input_text}
    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()
    print(output)
    inner_list = output[0]
    filtered_output = {map_labels(entry['label']): entry['score'] for entry in inner_list if entry['label'] in LABEL_MAP or entry['label'] in LABELS}
    
    return filtered_output
