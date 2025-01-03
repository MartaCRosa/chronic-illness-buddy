from openai import OpenAI
client = OpenAI(api_key="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")

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Say hi"
        }
    ],
    max_tokens=10
)

print(completion.choices[0].message)