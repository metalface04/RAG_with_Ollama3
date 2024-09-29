import requests

url = "https://api.sarvam.ai/text-to-speech"

payload = {
    "target_language_code": "en-IN",
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.65,
    "loudness": 1,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1",
    "inputs": ["hello"]
}
headers = {
    "api-subscription-key": "35cfb45a-6116-4157-b9bc-78b32b0cf87a",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)