import os
import requests

OPENROUTER_API_KEY = "AIzaSyBQhoJhC7JpDdiKGHhjz3HP45uoeGDmIHE"
url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gemini-2.5-flash",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "response_format": {"type": "json_object"}
}

try:
    print("Making request...")
    response = requests.post(url, headers=headers, json=data)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
except Exception as e:
    print("Exception:", e)
