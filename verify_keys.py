import os
import sys
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq

def verify_hf(token):
    masked = f"{token[:4]}...{token[-4:]}" if token and len(token) > 8 else "None"
    print(f"\nTesting HuggingFace Token: {masked}")
    
    if not token:
        print("❌ Error: HF_TOKEN is empty or missing.")
        return False
        
    try:
        client = InferenceClient(token=token)
        # Test 1: Check Token Validity with a standard/reliable model
        try:
            client.text_generation(prompt="Test", model="openai-community/gpt2", max_new_tokens=1)
            print("✅ HuggingFace Token: Valid (Connected via gpt2)")
            token_valid = True
        except Exception as e:
             if "401" in str(e):
                 print("❌ HuggingFace Token: Invalid (401 Unauthorized)")
                 return False
             print(f"⚠️ HuggingFace Token: Status Unsure (gpt2 check failed: {e})")
             token_valid = False

        # Test 2: Check Project Model Availability
        try:
            client.text_generation(prompt="Test", model="HuggingFaceH4/zephyr-7b-beta", max_new_tokens=1)
            print("✅ Project Model (Zephyr): Available")
        except Exception as e:
            print(f"⚠️ Project Model (Zephyr): Issue Detected - {e}")
            print("   (The app may fallback to default behaviors due to this)")
            
        return token_valid
    except Exception as e:
        print(f"❌ HuggingFace API: Connection Failed - {e}")
        return False

def verify_groq(key):
    masked = f"{key[:4]}...{key[-4:]}" if key and len(key) > 8 else "None"
    print(f"\nTesting Groq API Key: {masked}")
    
    if not key:
        print("❌ Error: GROQ_API_KEY is empty or missing.")
        return False
        
    try:
        client = Groq(api_key=key)
        # Simple test call
        # Use a very small model or the one we configured
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        print("✅ Groq API: Connected Successfully")
        return True
    except Exception as e:
        print(f"❌ Groq API: Failed")
        print(f"   Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Force reload of .env
    load_dotenv(override=True)
    
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Env file exists: {os.path.exists('.env')}")
    
    hf_token = os.getenv("HF_TOKEN")
    groq_key = os.getenv("GROQ_API_KEY")
    
    hf_status = verify_hf(hf_token)
    groq_status = verify_groq(groq_key)
    
    if not hf_status and not groq_status:
        print("\n⚠️  WARNING: Both API checks failed.")
        sys.exit(1)
    
    print("\nVerification Complete.")
