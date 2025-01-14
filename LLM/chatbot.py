import os
from openai import OpenAI

# Load the API key from an environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# Initial context and memory setup
context = "You are a helpful assistant giving medical advice. Your patient is diabetic. Answer concisely and clearly."
memory = [{"role": "system", "content": context}]  # Memory starts with the system context

def generate_response(user_input):
    # Append user input to memory
    memory.append({"role": "user", "content": user_input})

    # Call the API with the updated memory
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=memory,
        temperature=0.1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the response content
    assistant_response = response.choices[0].message.content

    # Append the assistant's response to memory
    memory.append({"role": "assistant", "content": assistant_response})

    return assistant_response
