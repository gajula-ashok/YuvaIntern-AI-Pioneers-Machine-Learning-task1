import requests
import json

url = 'http://127.0.0'
# Sending sample payload: 8 Years Experience, 85 Test Score, 3 Certifications
payload = {"features": [8.0, 85.0, 3.0]}

try:
    response = requests.post(url, json=payload)
    print("--- Step 3: API Production Test Result ---")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
except requests.exceptions.ConnectionError:
    print("Connection Error: Make sure app.py is running on http://127.0.0.1:5000")
