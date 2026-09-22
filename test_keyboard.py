import time
import requests

# 1. Define your FastAPI URL (change the port/IP if your server runs elsewhere)
API_URL = "https://1e76-113-140-3-91.ngrok-free.app/predict" 

# 2. Define your evaluation test dataset (Context -> Expected next word)
# 2. Define your evaluation test dataset (Context -> Expected next word)
TEST_CASES = [
    # English Tests (Notice the space at the very end of the context string!)
    {"context": "I want to go to the ", "expected": "market", "lang": "en"},
    {"context": "Please turn off the ", "expected": "lights", "lang": "en"},
    
    # French Tests (Notice the space at the very end!)
    {"context": "Je vais manger une ", "expected": "pomme", "lang": "fr"},
    {"context": "S'il vous ", "expected": "plaît", "lang": "fr"},
    
    # Chinese Tests (No space needed for Chinese characters)
    {"context": "今天天气很", "expected": "好", "lang": "zh"},
    {"context": "我想吃中国", "expected": "菜", "lang": "zh"}
]

def run_automated_test():
    total_tests = len(TEST_CASES)
    successful_predictions = 0
    total_latency = 0.0

    print("🚀 Starting Automated Keyboard Backend Test...\n")
    print(f"{'Language':<10}{'Input Context':<25}{'Expected':<12}{'Match?':<10}{'Latency':<10}")
    print("-" * 70)

    for case in TEST_CASES:
        payload = {
            "text": case["context"],
            "lang": case["lang"]
        }
        
        # Track the time right before sending the API request
        start_time = time.time()
        
        try:
            # Send request to your FastAPI backend
            response = requests.post(API_URL, json=payload, timeout=5)
            # Track time immediately when the response hits your Mac
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            total_latency += latency

            if response.status_code == 200:
                # Assuming your API returns a list of strings: {"predictions": ["word1", "word2", "word3"]}
                predictions = response.json().get("predictions", [])
                
                # Check if the expected word is anywhere in the top predictions
                if case["expected"].lower() in [p.lower() for p in predictions]:
                    match_status = "✅ YES"
                    successful_predictions += 1
                else:
                    match_status = "❌ NO"
                    
                print(f"{case['lang']:<10}{case['context']:<25}{case['expected']:<12}{match_status:<10}{latency:.1f}ms | Returned: {predictions}")
            else:
                print(f"{case['lang']:<10} Error: Status Code {response.status_code}")
                
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return

    # 3. Calculate and display final thesis metrics
    accuracy = (successful_predictions / total_tests) * 100
    avg_latency = total_latency / total_tests

    print("-" * 70)
    print("📊 FINAL EVALUATION METRICS FOR THESIS:")
    print(f"• Total Test Cases Evaluated: {total_tests}")
    print(f"• Next-Word Suggestion Accuracy: {accuracy:.2f}%")
    print(f"• Average Response Latency: {avg_latency:.1f} ms")

if __name__ == "__main__":
    run_automated_test()