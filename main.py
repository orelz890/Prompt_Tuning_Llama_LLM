from configuration.config import Config

conf = Config()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.prompt_tuning_pipeline import PromptTuningPipeline
from utils.Aviman1DatasetProcessor import Aviman1DatasetProcessor
from utils.GooglePersonaDatasetProcessor import GooglePersonaDatasetProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, BlenderbotForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import TaskType


actions = {'exit': 0, 'infer': 1, 'train': 2, 'visualize': 3}
foundational_models = {'facebook_400M': 1, 'other': 2}

from transformers import BlenderbotForConditionalGeneration

class BlenderbotWithEmbeds(BlenderbotForConditionalGeneration):
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds`.")
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

def get_user_action(instruction, options = None, type = int):
    if options:
        print(options)
    
    return type(input(f"[{instruction}]: "))

def get_required_info_from_user():
    # action = get_user_action(options="Actions: exit: 0, infer: 1, train: 2, visualize: 3", instruction="Enter Action Number", type=int)
    
    # # Exit
    # if action == actions['exit']:
    #     exit(0)
        
    # # User Input - Output Folder 
    # prompt_model_name = get_user_action(instruction="Enter your prompt model name", type=str)
    
    # # Foundational Model
    # foundational_model_name = get_user_action(instruction="Enter the foundational model path like: facebook/blenderbot-400M-distill", type=str)
    
    # dataset_name = get_user_action(instruction="Enter the dataset path like: google/Synthetic-Persona-Chat", type=str)
    
    # return action, prompt_model_name.lower(), foundational_model_name, dataset_name

    model_name = "llama_avi_1b_64vt_50e_16"
    # return 1, model_name, "facebook/blenderbot-400M-distill", "google/Synthetic-Persona-Chat"
    # return 2, model_name, "facebook/blenderbot-400M-distill", "google/Synthetic-Persona-Chat"
    # return 3, model_name, "facebook/blenderbot-400M-distill", "google/Synthetic-Persona-Chat"
    
    # return 1, model_name, "facebook/blenderbot-400M-distill", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
    # return 2, model_name, "facebook/blenderbot-400M-distill", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
    # return 3, model_name, "facebook/blenderbot-400M-distill", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
    
    # return 1, model_name, "unsloth/Llama-3.2-1B-Instruct", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
    # return 2, model_name, "unsloth/Llama-3.2-1B-Instruct", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"
    return 3, model_name, "unsloth/Llama-3.2-1B-Instruct", "Aviman1/Bot_Human_Prompt_Tuning_Dataset"

def get_auto_model_for_specific_llm(foundational_model_name):
    taskType = TaskType.CAUSAL_LM
    model_loader = AutoModelForCausalLM
    
    if foundational_model_name == "facebook/blenderbot-400M-distill":
        print("Using: BlenderbotWithEmbeds")
        taskType = TaskType.SEQ_2_SEQ_LM
        model_loader = BlenderbotWithEmbeds
    elif foundational_model_name == "facebook/blenderbot-400M-distill":
        taskType = TaskType.SEQ_2_SEQ_LM

    return model_loader, taskType


def main():
    
    flag = True
    while flag:
        flag = False
        
        try:
            action, prompt_model_name, foundational_model_name, dataset_name = get_required_info_from_user()
            
            # Foundational Model Folder Path
            local_model_dir = os.path.join(conf.PATHS["base_local_model_dir"], foundational_model_name.lower())

            # Output Dir Path
            output_dir = os.path.join(conf.PATHS["base_output_dir"], foundational_model_name.lower(), prompt_model_name.lower())

            model_loader, task_type = get_auto_model_for_specific_llm(foundational_model_name)
            
            pipeline: PromptTuningPipeline = PromptTuningPipeline(
                model_name = foundational_model_name,
                dataset_path = dataset_name,
                output_dir = output_dir,
                local_model_dir = local_model_dir,
                device = conf.DEVICE,
                auto_tokenizer = AutoTokenizer,
                model_loader = model_loader,
                task_type = task_type
            )
            
            if action == actions['train']:
                # # Check if the name already exist
                # if os.path.isdir(output_dir):
                #     print(f"The directory {output_dir} already exists.")
                #     continue
                
                pipeline.train(
                    # Change Default Values If Needed. Example:
                    # epochs=5
                    # dataset_processor = Aviman1DatasetProcessor
                    # task_type = TaskType.SEQ_2_SEQ_LM
                )

            elif action == actions['infer']:
                
                pipeline.infer(
                    # Change Default Values If Needed
                )
            elif action == actions['visualize']:
                pipeline.visualize(output_dir)
        
        except ValueError as e:
            print(e)
    

if __name__ == "__main__":
    main()


""" 
    TODO:
    1.  [-] Understand why it always generate the max amount - padding data? mask?
    2.  [v] Probably related, fix the warning "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation."
    3.  [v] Fix GPU out of memory.
    4.  [v] Talk with Avi about the dataset - get a better one.
    5.  [-] Complete the other strategies.
    6.  [v] Test the model before training.
    7.  [v] Test the model after small amount of epochs training.
    8.  [-] Ask Amos if we want a lot of epochs - intentional overfiting?
    9.  [v] Find a smaller chat bot model.
    10. [-] Dive into the Prompt Engineering part - Read about it and how to use peft lib.
    
    
    11. why is he repeating the input?
    12. Try to use a more suitable model?
     
""" 
