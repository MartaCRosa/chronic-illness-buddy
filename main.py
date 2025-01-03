from LLM.openai import generate_response

user_input = "I'm worried I could be hyperglycemic, what are some symptoms that could indicate that I am?"
response = generate_response(user_input)
print(response)