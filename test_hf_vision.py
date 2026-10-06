import glob
import requests

keyframes = glob.glob("static/keyframes/*.jpg")
print(f"Found {len(keyframes)} keyframe images.")

if keyframes:
    img_path = keyframes[0]
    print(f"Testing free HF vision API on: {img_path}")
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # Hugging Face Free Inference Endpoint for BLIP Vision model
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    
    try:
        response = requests.post(api_url, data=img_bytes, timeout=15)
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as e:
        print("HF Vision Error:", e)
