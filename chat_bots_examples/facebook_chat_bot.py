import os
import gc
from transformers import BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Clear GPU Cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.memory_stats()
    print(torch.cuda.memory_summary(device="cuda"))

# Model configuration
model_name = "facebook/blenderbot-400M-distill"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Main function for interactive chat
def main():
    print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

    # Conversation history
    conversation_history = ""

    while True:
        try:
            # Get user input
            user_input = input("[USER]: ")
            if user_input.lower() in {"exit", "quit"}:
                print("[INFO] Exiting chat.")
                break

            # Append user input to conversation history with correct formatting
            if conversation_history:
                conversation_history += f" </s> <s>{user_input}"
            else:
                conversation_history = f"<s>{user_input}"

            # Tokenize the updated conversation history
            inputs = tokenizer(
                [conversation_history],
                return_tensors="pt",
                truncation=True,
                max_length=256  # Limit the length to avoid context overflow
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate model response
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                do_sample=True,             # Enable sampling for varied responses
                # temperature=0.7,            # Control randomness
                top_p=0.9,                  # Nucleus sampling
                top_k=50,                   # Top-K sampling
                max_new_tokens=50,          # Limit generated tokens
                repetition_penalty=1.2,     # Penalize repetition
            )

            # Decode and display the response
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"[MODEL]: {response}")

            # Update the conversation history with the bot's response
            conversation_history += f" </s> <s>{response}"

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            break


if __name__ == "__main__":
    gc.collect()  # Clean up memory
    main()
