import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please set DEEPSEEK_API_KEY in your .env file")

# Initialize the OpenAI client with DeepSeek's base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

def chat_with_deepseek(user_message, system_message=None, model="deepseek-chat", temperature=0.7):
    """
    Send a message to DeepSeek AI and get a response
    
    Args:
        user_message (str): The user's message
        system_message (str, optional): System message to set context
        model (str): Model to use (deepseek-chat or deepseek-coder)
        temperature (float): Controls randomness (0-1)
    
    Returns:
        str: AI's response
    """
    try:
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000  # Adjust as needed
        )
        
        # Extract and return the response
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"

def stream_chat_with_deepseek(user_message, system_message=None, model="deepseek-chat"):
    """
    Stream the response from DeepSeek AI token by token
    """
    try:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        # Stream the response
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print()  # New line after streaming
        return full_response
        
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Simple chat example
    print("Simple Chat Example:")
    response = chat_with_deepseek(
        "What is artificial intelligence?",
        system_message="You are a helpful assistant that explains complex topics simply."
    )
    print(response)
    
    print("\n" + "="*50 + "\n")
    
    # Streaming example
    print("Streaming Example:")
    stream_chat_with_deepseek(
        "Tell me a short joke.",
        system_message="You are a funny assistant."
    )
    
    print("\n" + "="*50 + "\n")
    
    # Interactive chat loop
    print("Interactive Chat (type 'quit' to exit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        response = chat_with_deepseek(user_input)
        print(f"AI: {response}")