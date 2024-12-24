import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Clear GPU Cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device="cuda"))

# Model configuration
model_name = "unsloth/Llama-3.2-1B-Instruct"  # Replace with your model name if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Ensure the tokenizer has padding tokens for input and output
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[INFO] Padding token set to: {tokenizer.pad_token}")


# Main function for interactive chat
def main():
    print("[INFO] Entering interactive chat mode. Type 'exit' or 'quit' to end the session.")

    # Initialize tokenized conversation history
    tokenized_conversation_history = None

    while True:
        try:
            # Get user input
            user_input = input("\n[USER]: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("[INFO] Exiting chat.")
                break

            # Add the user's input with proper prefix
            user_input_formatted = f"[USER]: {user_input}\n[MODEL]: "

            # Tokenize the user's input
            user_input_tokenized = tokenizer(
                user_input_formatted,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # Adjust max length if needed
            ).to(device)

            # Concatenate tokenized conversation history with new input
            if tokenized_conversation_history is None:
                # First turn: only use the new input
                combined_inputs = user_input_tokenized
            else:
                # Subsequent turns: concatenate history with new input
                combined_inputs = {
                    "input_ids": torch.cat(
                        [tokenized_conversation_history["input_ids"], user_input_tokenized["input_ids"]], dim=1
                    ),
                    "attention_mask": torch.cat(
                        [tokenized_conversation_history["attention_mask"], user_input_tokenized["attention_mask"]], dim=1
                    ),
                }

            # Generate the model's response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=combined_inputs["input_ids"],
                    attention_mask=combined_inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.7,  # Control randomness
                    top_p=0.9,  # Nucleus sampling
                    top_k=50,  # Top-K sampling
                    max_new_tokens=50,  # Limit number of new tokens
                    repetition_penalty=1.2,  # Penalize repetition
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract the newly generated tokens
            input_length = combined_inputs["input_ids"].shape[1]
            new_tokens = outputs[0, input_length:]

            # Decode only the new tokens
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"[MODEL]: {response}")

            # Update the tokenized conversation history without including `[MODEL]:`
            updated_response_tokenized = tokenizer(
                f"{response}\n",
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            tokenized_conversation_history = {
                "input_ids": torch.cat(
                    [combined_inputs["input_ids"], updated_response_tokenized["input_ids"]], dim=1
                ),
                "attention_mask": torch.cat(
                    [combined_inputs["attention_mask"], updated_response_tokenized["attention_mask"]], dim=1
                ),
            }

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            break


if __name__ == "__main__":
    gc.collect()  # Clean up memory
    main()
