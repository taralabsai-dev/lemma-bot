#!/usr/bin/env python3
"""
Ollama Setup and Test Script for macOS
Optimized for Apple Silicon (M4 Mini)
"""

import subprocess
import sys
import time
import json
import platform
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class OllamaSetup:
    def __init__(self):
        self.ollama_api_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        self.is_apple_silicon = platform.machine() == "arm64"
        
    def check_ollama_installed(self):
        """Check if Ollama is installed on the system."""
        try:
            result = subprocess.run(["which", "ollama"], 
                                  capture_output=True, 
                                  text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def check_ollama_running(self):
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.ollama_api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama_service(self):
        """Start Ollama service."""
        print("Starting Ollama service...")
        try:
            # Start Ollama in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(10):
                time.sleep(2)
                if self.check_ollama_running():
                    print("✅ Ollama service started successfully!")
                    return True
                print(f"Waiting for Ollama to start... ({i+1}/10)")
            
            return False
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return False
    
    def install_instructions(self):
        """Display installation instructions."""
        print("\n" + "="*60)
        print("🚀 OLLAMA INSTALLATION INSTRUCTIONS")
        print("="*60)
        print("\nOllama is not installed. Please install it using one of these methods:\n")
        
        print("Option 1: Install via Homebrew (Recommended)")
        print("-" * 40)
        print("brew install ollama")
        print("\nOption 2: Download from official website")
        print("-" * 40)
        print("Visit: https://ollama.ai/download")
        print("\n" + "="*60 + "\n")
        
        if self.is_apple_silicon:
            self.show_m4_optimization()
    
    def show_m4_optimization(self):
        """Display M4 Mini optimization settings."""
        print("\n" + "="*60)
        print("🎯 APPLE SILICON (M4 MINI) OPTIMIZATION")
        print("="*60)
        print("\nFor optimal performance on M4 Mini, configure these settings:\n")
        
        print("1. Memory allocation (add to ~/.zshrc or ~/.bash_profile):")
        print("-" * 40)
        print("export OLLAMA_NUM_GPU=1")
        print("export OLLAMA_GPU_MEMORY_FRACTION=0.9")
        print("export OLLAMA_MAX_LOADED_MODELS=1")
        print()
        
        print("2. Model-specific optimizations:")
        print("-" * 40)
        print("# For better performance, use quantized models:")
        print("ollama pull llama3.2:3b-q4_K_M  # 4-bit quantization")
        print("ollama pull llama3.2:3b-q8_0   # 8-bit quantization")
        print()
        
        print("3. Ollama configuration (~/.ollama/config.json):")
        print("-" * 40)
        config_example = {
            "gpu": {
                "enabled": True,
                "memory_fraction": 0.9
            },
            "models": {
                "options": {
                    "num_gpu": 1,
                    "num_thread": 8,
                    "f16_kv": True
                }
            }
        }
        print(json.dumps(config_example, indent=2))
        print()
        
        print("4. System settings for optimal performance:")
        print("-" * 40)
        print("- Close unnecessary applications")
        print("- Disable Spotlight indexing for Ollama directories")
        print("- Use Activity Monitor to ensure sufficient RAM available")
        print("- Consider using 'Low Power Mode' OFF for max performance")
        print("\n" + "="*60 + "\n")
    
    def check_model_exists(self):
        """Check if the model is already downloaded."""
        try:
            response = requests.get(f"{self.ollama_api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model_name for model in models)
        except:
            return False
        return False
    
    def pull_model(self):
        """Pull the llama3.2:3b model."""
        print(f"\n📥 Pulling {self.model_name} model...")
        print("This may take several minutes depending on your internet speed...\n")
        
        try:
            # Use subprocess to show real-time progress
            process = subprocess.Popen(
                ["ollama", "pull", self.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✅ Successfully pulled {self.model_name}!")
                return True
            else:
                print(f"\n❌ Failed to pull {self.model_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
    
    def test_llm(self):
        """Test the LLM with a simple prompt."""
        print("\n🧪 Testing LLM with a simple prompt...")
        print("-" * 40)
        
        test_prompt = "What is the capital of France? Please answer in one sentence."
        
        try:
            response = requests.post(
                f"{self.ollama_api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 100
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prompt: {test_prompt}")
                print(f"Response: {result.get('response', 'No response')}")
                print("\n✅ LLM test successful!")
                return True
            else:
                print(f"❌ Test failed with status code: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Test timed out. The model might be loading for the first time.")
            print("   Please try again in a moment.")
            return False
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False
    
    def advanced_test(self):
        """Run an advanced test relevant to trading."""
        print("\n🚀 Running advanced trading-related test...")
        print("-" * 40)
        
        trading_prompt = """You are a financial analyst. Given these metrics for AAPL:
- Current Price: $185.50
- 50-day SMA: $182.30
- RSI: 65
- Volume: Above average

Provide a one-sentence trading recommendation."""
        
        try:
            response = requests.post(
                f"{self.ollama_api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": trading_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent analysis
                        "top_p": 0.9,
                        "max_tokens": 150
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("Trading Analysis Test:")
                print(f"Response: {result.get('response', 'No response')}")
                
                # Test response time
                total_time = result.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
                print(f"\n⏱️  Response time: {total_time:.2f} seconds")
                
                if self.is_apple_silicon:
                    tokens_per_second = result.get('eval_count', 0) / (result.get('eval_duration', 1) / 1e9)
                    print(f"🚀 Tokens/second: {tokens_per_second:.2f}")
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Advanced test error: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🔧 Ollama Setup Script for macOS")
        print("=" * 60)
        
        # Step 1: Check if Ollama is installed
        if not self.check_ollama_installed():
            self.install_instructions()
            print("\n⚠️  Please install Ollama and run this script again.")
            return False
        
        print("✅ Ollama is installed")
        
        # Step 2: Check if Ollama service is running
        if not self.check_ollama_running():
            print("⚠️  Ollama service is not running")
            if not self.start_ollama_service():
                print("\n❌ Failed to start Ollama service")
                print("Try starting it manually with: ollama serve")
                return False
        else:
            print("✅ Ollama service is running")
        
        # Step 3: Check if model exists, pull if needed
        if not self.check_model_exists():
            print(f"⚠️  Model {self.model_name} not found")
            if not self.pull_model():
                return False
        else:
            print(f"✅ Model {self.model_name} is available")
        
        # Step 4: Test the LLM
        if not self.test_llm():
            return False
        
        # Step 5: Run advanced test
        self.advanced_test()
        
        # Show optimization tips for Apple Silicon
        if self.is_apple_silicon:
            print("\n💡 Tip: Review the M4 Mini optimization settings above for better performance")
        
        print("\n🎉 Ollama setup completed successfully!")
        print(f"📍 API URL: {self.ollama_api_url}")
        print(f"🤖 Model: {self.model_name}")
        
        return True


def main():
    """Main function."""
    setup = OllamaSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()