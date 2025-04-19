from openai import AzureOpenAI

# Correct Azure OpenAI resource details
client = AzureOpenAI(
    api_key="your_model_key",  # Azure OpenAI Key
    azure_endpoint="you_cognitive_service_api",  # Correct endpoint format
    api_version="2024-12-01-preview"
)

# Chat completion call
response = client.chat.completions.create(
    model="your_model_name",  # Deployment name, NOT model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am going to Paris, what should I see?"}
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0
)

# Print the assistant's reply
print(response.choices[0].message.content)
