from openai import OpenAI

client = OpenAI(api_key="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")

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
