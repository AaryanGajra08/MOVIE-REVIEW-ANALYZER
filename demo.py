import requests

API = "http://127.0.0.1:8000"

reviews = [
    "The acting was excellent and the movie was fantastic.",
    "The story was boring and disappointing."
]

for review in reviews:
    response = requests.post(f"{API}/predict", json={"review": review})
    print(response.json())
