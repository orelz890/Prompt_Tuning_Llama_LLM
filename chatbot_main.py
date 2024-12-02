

from transformers import AutoModelForCausalLM, AutoTokenizer
import os


base_local_model_dir = "./local_model"

# model_name = "meta-llama/Llama-3.1-8b-instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

local_model_dir = os.path.join(base_local_model_dir, model_name.lower())

tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(local_model_dir)


# Ensure the tokenizer has padding tokens for input and output
if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("[PAD]")
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if new tokens are added
    print(f"[INFO] Padding token set to: {tokenizer.pad_token}")
            
# Set the models pad token id
model.generation_config.pad_token_id = tokenizer.pad_token_id


def main():
    
    print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

    while True:
        user_input = input("[USER]: ")
        if user_input.lower() in {"exit", "quit"}:
            print("[INFO] Exiting chat.")
            break


        # Tokenize user input with attention mask
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Cap input length
        )


        # Generate output with controlled length and sampling
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Provide attention mask
            do_sample=True,
            temperature=0.7,  # Adjust temperature for controlled randomness
            max_new_tokens=30,
            # repetition_penalty=1.5, # Discourage repeating phrases
            # length_penalty=1.5, # Penalize very long outputs
            top_k=40,
            top_p=0.9,
        )

        # Decode and print the model's response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[MODEL]: {response}")

if __name__ == "__main__":
    main()
    