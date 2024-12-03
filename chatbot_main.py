import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import gc

gc.collect()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

if torch.cuda.is_available():
    # Clear cached data
    torch.cuda.empty_cache()
    
    # Force PyTorch to release cached memory
    torch.cuda.memory_stats()

    # torch.cuda.set_per_process_memory_fraction(0.2)
    print(torch.cuda.memory_summary(device="cuda"))
    device = 'cuda'
    
base_local_model_dir = "./local_model"

# model_name = "meta-llama/Llama-3.1-8b-instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# model_name = "cmarkea/bloomz-560m-sft-chat"


local_model_dir = os.path.join(base_local_model_dir, model_name.lower())

tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(local_model_dir).to(device)


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
            max_length=512  # Cap input length
        )

        # Move inputs to the GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate output with controlled length and sampling
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Provide attention mask
            do_sample=True,
            temperature=0.25,  # Adjust temperature for controlled randomness
            max_new_tokens=50,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.5,  # Avoid repetition.
            # length_penalty=1.5, # Penalize very long outputs
            # early_stopping=True,  # The model can stop before reach the max_length
            eos_token_id=tokenizer.eos_token_id,
        )

        # Count total tokens in the generated output
        total_tokens = outputs.size(1)  # Outputs is of shape (batch_size, seq_length)

        # Count input tokens (optional, if you want new tokens only)
        input_token_count = inputs["input_ids"].size(1)

        # Count only newly generated tokens
        new_tokens_count = total_tokens - input_token_count

        print(f"Total tokens in output: {total_tokens}")
        print(f"Input tokens: {input_token_count}")
        print(f"Newly generated tokens: {new_tokens_count}")
                
        # Decode and print the model's response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[MODEL]: {response}")

if __name__ == "__main__":
    main()
    