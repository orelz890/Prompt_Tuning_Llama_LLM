import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Suppress the position IDs warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids."
)


# # Inference Strategy
# class InferenceStrategy(BasePipelineStrategy):
#     def __init__(self, model_manager: ModelManager):
#         self.model_manager = model_manager

#     def execute(self):
#         print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")
#         while True:
#             user_input = input("[USER]: ")
#             if user_input.lower() in {"exit", "quit"}:
#                 print("[INFO] Exiting chat.")
#                 break

#             inputs = self.model_manager.tokenizer(user_input, return_tensors="pt")
#             outputs = self.model_manager.model.generate(
#                 input_ids=inputs["input_ids"], do_sample=True, temperature=0.1
#             )
#             print(f"[MODEL]: {self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)}")


# Inference Strategy
class InferenceStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        
        # Ensure `pad_token_id` is set to a valid token
        self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token or "[PAD]"
        self.model_manager.tokenizer.pad_token_id = self.model_manager.tokenizer.eos_token_id or self.model_manager.tokenizer.convert_tokens_to_ids("[PAD]")

        # Update the model configuration
        self.model_manager.model.config.pad_token_id = self.model_manager.tokenizer.pad_token_id
        self.model_manager.model.config.eos_token_id = self.model_manager.tokenizer.eos_token_id

    def execute(self, max_length, temperature, max_new_tokens, top_k, top_p):
        print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

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

            # Decode and print the model's response
            response = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[MODEL]: {response}")
