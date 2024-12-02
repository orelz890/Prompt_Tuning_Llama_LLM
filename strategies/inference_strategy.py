import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Suppress the position IDs warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids."
)

# Inference Strategy
class InferenceStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        
        # Set the models pad token id
        self.model_manager.model.generation_config.pad_token_id = self.model_manager.tokenizer.pad_token_id

    def execute(self, max_length, temperature, max_new_tokens, top_k, top_p):
        print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

        with torch.no_grad():
            while True:
                user_input = input("[USER]: ")
                if user_input.lower() in {"exit", "quit"}:
                    print("[INFO] Exiting chat.")
                    break

                
                # Tokenize user input with attention mask
                inputs = self.model_manager.tokenizer(
                    user_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length  # Cap input length
                ).to(self.device)


                # Generate output with controlled length and sampling
                outputs = self.model_manager.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],  # Provide attention mask
                    do_sample=True,
                    temperature=temperature,  # Adjust temperature for controlled randomness
                    max_new_tokens=max_new_tokens,
                    # repetition_penalty=1.5, # Discourage repeating phrases
                    # length_penalty=1.5, # Penalize very long outputs
                    top_k=top_k,
                    top_p=top_p
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
                response = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"[MODEL]: {response}")