from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
