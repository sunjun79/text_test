import requests

def call_local_llm(prompt: str, model: str = "gemma2:2b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1024,
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = response.json()
    return data.get("response", "")
