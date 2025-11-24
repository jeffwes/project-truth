import os
import requests
import sys

def test_gemini_3():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # 1. The official Model ID for Gemini 3
    model_name = "gemini-3-pro-preview"

    # 2. The correct URL (uses 'v1beta' and ':generateContent')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    # 3. The required JSON payload structure
    payload = {
        "contents": [{
            "parts": [{"text": "Are you the Gemini 3 model?"}]
        }]
    }

    print(f"Testing access to {model_name}...")
    
    try:
        response = requests.post(
            url, 
            params={"key": api_key}, 
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            # Extract the text from the response
            try:
                answer = data['candidates'][0]['content']['parts'][0]['text']
                print(f"\n[SUCCESS] Connected to {model_name}!")
                print(f"Response: {answer}")
            except KeyError:
                print(f"\n[WARNING] Connected, but unexpected JSON format: {data}")
        else:
            print(f"\n[FAILURE] Status Code: {response.status_code}")
            print(f"Error Message: {response.text}")
            
            if response.status_code == 404:
                print("\nNOTE: A 404 error often means your API Key does not have access to 'Preview' models.")
                print("Check if you need to enable billing or join the waitlist in Google AI Studio.")

    except Exception as e:
        print(f"Network error: {e}")

if __name__ == "__main__":
    test_gemini_3()
