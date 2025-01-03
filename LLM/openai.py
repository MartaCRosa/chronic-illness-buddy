from openai import OpenAI

client = OpenAI(api_key="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")

context = "You are a helpful assistant giving medical advice. Your patient is diabetic. Answer concisely and clearly." #Add things to this

def generate_response(user_input):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": context},
                {"role": "user", "content": user_input}
    ],
    response_format={
        "type": "text"
    },
    temperature=0.1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].message.content