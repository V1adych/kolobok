# test.py

import os
import base64
import requests


API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN not set in environment")

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
URL = "http://localhost:8000"

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def post(endpoint: str, img_b64: str):
    resp = requests.post(
        f"{URL}{endpoint}",
        json={"image": img_b64},
        headers=HEADERS,
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()

def main():
    img_b64 = encode_image("image.png")

    print("Calling /api/v1/analyze_thread …")
    out1 = post("/api/v1/analyze_thread", img_b64)
    print(out1, end="\n\n")

    print("Calling /api/v1/identify_tire …")
    out2 = post("/api/v1/identify_tire", img_b64)
    print(out2)

if __name__ == "__main__":
    main()