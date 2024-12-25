import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainerCallback
import torch
from transformers import pipeline

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Debugging Strategy
class DebuggingStrategy(BasePipelineStrategy, TrainerCallback):
    """
    A strategy for debugging model outputs at the end of each training epoch.
    Integrates with the Hugging Face Trainer as a callback.

    Attributes:
        model: The model being trained or debugged.
        tokenizer: Tokenizer used for encoding inputs and decoding outputs.
        device (str): The device (e.g., 'cuda', 'cpu') used for computations.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the DebuggingStrategy with the model, tokenizer, and device.

        Args:
            model: The model to debug.
            tokenizer: Tokenizer for processing inputs and outputs.
            device (str): Device to run computations ('cuda' or 'cpu').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Executes the debugging process at the end of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control object.
            **kwargs: Additional arguments.
        """
        self.execute()

    def execute(self):
        """
        Perform model debugging by generating outputs for predefined test inputs.
        """
        self.model.eval()
        print("\n[INFO] Debugging model output...")

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # Predefined test inputs
        test_inputs = ["how are you?", "HOW ARE YOU", "what are you doing?", "where are you from", "Are you a bot?"]

        for input_text in test_inputs:
            # Format the input as an instruct-style message
            messages = [{"role": "user", "content": input_text}]

            # Generate response using the pipeline
            response = pipe(messages, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)

            # Display the response
            print(f"[MODEL]: {response[0]['generated_text']}")

        self.model.train()
        print("\n[INFO] Finished Debugging model output...")


# ===========================

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from transformers import TrainerCallback
# import torch
# from transformers import pipeline

# from strategies.base_strategy import BasePipelineStrategy
# from managers.model_manager import ModelManager

# # Debugging Strategy
# class DebuggingStrategy(BasePipelineStrategy, TrainerCallback):
#     """
#     A strategy for debugging model outputs at the end of each training epoch.
#     Integrates with the Hugging Face Trainer as a callback.

#     Attributes:
#         model: The model being trained or debugged.
#         tokenizer: Tokenizer used for encoding inputs and decoding outputs.
#         device (str): The device (e.g., 'cuda', 'cpu') used for computations.
#     """
    
#     def __init__(self, model, tokenizer, device):
#         """
#         Initialize the DebuggingStrategy with the model, tokenizer, and device.

#         Args:
#             model: The model to debug.
#             tokenizer: Tokenizer for processing inputs and outputs.
#             device (str): Device to run computations ('cuda' or 'cpu').
#         """
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device

#     def on_epoch_end(self, args, state, control, **kwargs):
#         """
#         Executes the debugging process at the end of each epoch.

#         Args:
#             args: Training arguments.
#             state: Trainer state.
#             control: Trainer control object.
#             **kwargs: Additional arguments.
#         """
#         self.execute()

#     def execute(self):
#         """
#         Perform model debugging by generating outputs for predefined test inputs.
#         """
#         self.model.eval()
#         print("\n[INFO] Debugging model output...")

#         # Predefined test inputs
#         test_inputs = ["how are you?", "HOW ARE YOU", "what are you doing?", "where are you from", "Are you a bot?"]

#         # Initialize tokenized conversation history
#         tokenized_conversation_history = None

#         for input_text in test_inputs:
#             print(f"\n[USER]: {input_text}")

#             # Format the user input with the `[MODEL]:` prefix for generation
#             user_input_formatted = f"[USER]: {input_text}\n[MODEL]: "

#             # Tokenize the user input
#             user_input_tokenized = self.tokenizer(
#                 user_input_formatted,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=256,
#             ).to(self.device)

#             # Concatenate tokenized conversation history with the new user input
#             if tokenized_conversation_history is None:
#                 # First turn: only use the new input
#                 combined_inputs = user_input_tokenized
#             else:
#                 # Subsequent turns: concatenate history with the new input
#                 combined_inputs = {
#                     "input_ids": torch.cat(
#                         [tokenized_conversation_history["input_ids"], user_input_tokenized["input_ids"]], dim=1
#                     ),
#                     "attention_mask": torch.cat(
#                         [tokenized_conversation_history["attention_mask"], user_input_tokenized["attention_mask"]], dim=1
#                     ),
#                 }

#             # Generate the model's response
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     input_ids=combined_inputs["input_ids"],
#                     attention_mask=combined_inputs["attention_mask"],
#                     do_sample=True,
#                     temperature=0.7,
#                     top_p=0.9,
#                     top_k=50,
#                     max_new_tokens=50,
#                     repetition_penalty=1.2,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                 )

#             # Extract the newly generated tokens
#             input_length = combined_inputs["input_ids"].shape[1]
#             new_tokens = outputs[0, input_length:]

#             # Decode only the new tokens
#             response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
#             print(f"[MODEL]: {response}")

#             # Tokenize the model's response for appending to history
#             response_tokenized = self.tokenizer(
#                 f"{response}\n",  # Add newline for proper formatting in conversation
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=256,
#             ).to(self.device)

#             # Update the tokenized conversation history
#             if tokenized_conversation_history is None:
#                 tokenized_conversation_history = response_tokenized
#             else:
#                 tokenized_conversation_history = {
#                     "input_ids": torch.cat(
#                         [tokenized_conversation_history["input_ids"], response_tokenized["input_ids"]], dim=1
#                     ),
#                     "attention_mask": torch.cat(
#                         [tokenized_conversation_history["attention_mask"], response_tokenized["attention_mask"]], dim=1
#                     ),
#                 }

#         self.model.train()
#         print("\n[INFO] Finished Debugging model output...")


# =======================

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# from transformers import TrainerCallback
# import torch

# from strategies.base_strategy import BasePipelineStrategy
# from managers.model_manager import ModelManager


# # Debugging Strategy
# class DebuggingStrategy(BasePipelineStrategy, TrainerCallback):
#     """
#     A strategy for debugging model outputs at the end of each training epoch.
#     Integrates with the Hugging Face Trainer as a callback.

#     Attributes:
#         model: The model being trained or debugged.
#         tokenizer: Tokenizer used for encoding inputs and decoding outputs.
#         device (str): The device (e.g., 'cuda', 'cpu') used for computations.
#     """
    
#     def __init__(self, model, tokenizer, device):
#         """
#         Initialize the DebuggingStrategy with the model, tokenizer, and device.

#         Args:
#             model: The model to debug.
#             tokenizer: Tokenizer for processing inputs and outputs.
#             device (str): Device to run computations ('cuda' or 'cpu').
#         """
        
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device

#     def on_epoch_end(self, args, state, control, **kwargs):
#         """
#         Executes the debugging process at the end of each epoch.

#         Args:
#             args: Training arguments.
#             state: Trainer state.
#             control: Trainer control object.
#             **kwargs: Additional arguments.
#         """
        
#         self.execute()
        
#     def execute(self):
#         """
#         Perform model debugging by generating outputs for predefined test inputs.
#         """
#         self.model.eval()
#         print("\n[INFO] Debugging model output...")
#         test_inputs = ["how are you?", "HOW ARE YOU", "what are you doing?", "where are you from", "Are you a bot?"]

#         # Conversation history
#         conversation_history = ""

#         for input_text in test_inputs:
#             # Append user input to conversation history
#             conversation_history += f" {input_text}" if conversation_history else input_text

#             # Tokenize the conversation history
#             inputs = self.tokenizer(
#                 conversation_history,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=256
#             ).to(self.device)

#             print("[DEBUG] Tokenized Input:", inputs)

#             # Generate model response
#             with torch.no_grad():
#                 # Ensure only `input_ids` and `attention_mask` are passed
#                 outputs = self.model.generate(
#                     input_ids=inputs["input_ids"],        # Pass tokenized input IDs
#                     # attention_mask=inputs["attention_mask"],  # Pass attention mask
#                     do_sample=True,             # Enable sampling for varied responses
#                     temperature=0.7,            # Control randomness
#                     top_p=0.9,                  # Nucleus sampling
#                     top_k=50,                   # Top-K sampling
#                     max_new_tokens=50,          # Limit generated tokens
#                     repetition_penalty=1.2,     # Penalize repetition
#                     eos_token_id=self.tokenizer.eos_token_id,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                 )

#                 # Decode and display the response
#                 response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#                 print(f"[MODEL]: {response}")

#         self.model.train()
#         print("\n[INFO] Finished Debugging model output...")
