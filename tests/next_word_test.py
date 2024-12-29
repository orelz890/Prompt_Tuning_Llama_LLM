import unittest
import sys
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.prompt_tuning_pipeline import PromptTuningPipeline
from managers.model_manager import ModelManager

class TestPromptTuning(unittest.TestCase):
    
    def setUp(self):
        # Set the device to GPU if available, else fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        prompt_model_name = "llama_avi_1b_64vt_50e_5"
        
        # Model configuration
        foundational_model_name = "unsloth/Llama-3.2-1B-Instruct"
        dataset_name = "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
        base_output_dir = "./pretrained"
        local_model_dir = "./local_model"
        
        pretrained_model_name = os.path.join(base_output_dir, foundational_model_name.lower(), prompt_model_name.lower())


        self.model_manager = ModelManager(model_name = foundational_model_name, 
                                          device = self.device, 
                                          local_model_dir = local_model_dir,
                                          auto_tokenizer = AutoTokenizer,
                                          model_loader = AutoModelForCausalLM,
                                          task_type = TaskType.CAUSAL_LM,
                                          )
        
        # Load the pretrained model
        self.model_manager.load_prompt_tuned_model(output_dir=pretrained_model_name)
        self.tokenizer = self.model_manager.tokenizer

    def test_next_word(self):

        # Predefined test inputs
        test_inputs = [("This is melanie here","hi this is denise", "here"), 
                       ("how are you","i am", "fine"), 
                       ("wow nice to hear what kind of food do you cook?","i like to", "cook"), 
                       ("have you like pizza?","Yes I ", "like"), 
                       ("Wow great, you are a software engineer right?","yes, I love my", "job"),
                       ("are you doing good today?","I am fine, What are you", "doing"),
                       ("what jo you do","software", "developer"),
                       ("Fine, how about you?","I AM", "FINE"),
                       ("which is your hobbies?","watching movie", "and"),
                       ("Thank you. What are you up to John?","simply chat with", "you"),
                       ("favorite place", "what about", "u"),
                       ("hi james how r u", "i am also", "cool"),
                       ("i am fine what about you?", "reading", "books"),
                       ("good mrng","good", "morning"),
                       ("how is your day", "its", "good"),
                       ("Nice whats there special", "nothing", "special"),
                       ("What work", "DATA", "ANALIS"),
                       ]

        messages = [
            # {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
        ]
        
        foundational_counter = 0
        peft_counter = 0
        
        for input_text, assist ,label in test_inputs:
            messages.append({"role": "user", "content": str(input_text)})
            messages.append({"role": "assistant", "content": str(assist)})
            
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            
            inputs["input_ids"] = inputs["input_ids"][:,:-5]
            inputs["attention_mask"] = inputs["attention_mask"][:,:-5]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
            # print(inputs['input_ids']," = ", tokens) 
            
            
            foundational_outputs = self.model_manager.get_output("foundational", inputs)
            peft_outputs = self.model_manager.get_output("peft", inputs)

            input_size = len(inputs["input_ids"][0])
            
            f_predict = self.tokenizer.decode(foundational_outputs[0][input_size: input_size + 1], skip_special_tokens=True).strip()
            p_predict = self.tokenizer.decode(peft_outputs[0][input_size: input_size + 1], skip_special_tokens=True).strip()
            print("foundational: ", f_predict)
            print("peft: ", p_predict)
            print("label: ", label, "\n")
            
            if f_predict.startswith(label):
                foundational_counter += 1
            if p_predict.startswith(label):
                peft_counter += 1

        print("foundational_counter: ", foundational_counter)
        print("peft_counter: ", peft_counter)


if __name__ == '__main__':
    unittest.main()