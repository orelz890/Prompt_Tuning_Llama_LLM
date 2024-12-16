from configuration.config import Config

conf = Config()

from huggingface_hub import InferenceClient


client = InferenceClient(api_key=conf.API_KEYS.get("hugging_token"))

# messages = "\"The answer to the universe is\""
messages = "how are you?"

completion = client.chat.completions.create(
    model="facebook/blenderbot-400M-distill", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)