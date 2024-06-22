import json

def extract_json(json_path='./resources/models/template_model.json'):
    with open(json_path, 'r') as f:
        # Load JSON data from the file
        data = json.load(f)
    return data