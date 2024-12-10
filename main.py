from configuration.config import Config

conf = Config()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.prompt_tuning_pipeline import PromptTuningPipeline
from utils.Aviman1DatasetProcessor import Aviman1DatasetProcessor
from utils.GooglePersonaDatasetProcessor import GooglePersonaDatasetProcessor


def get_user_action(instruction, options = None, type = int):
    if options:
        print(options)
    
    return type(input(f"[{instruction}]: "))


def main():
    
    # Foundational Model Folder Path
    local_model_dir = os.path.join(conf.PATHS["base_local_model_dir"], conf.MODELS["foundational_model"].lower())
    
    actions = {'exit': 0, 'infer': 1, 'train': 2, 'visualize': 3}
    
    while True:
        
        try:
            action = get_user_action(options="Actions: exit: 0, infer: 1, train: 2, visualize: 3", instruction="Enter Action Number", type=int)
            
            # Exit
            if action == actions['exit']:
                return
            
            # Invalid Input
            if not isinstance(action, int) or action not in actions.values():
                print("Invalid action number.")
                continue
            
            # User Input - Output Folder 
            subfolder_name = get_user_action(instruction="Enter model name", type=str)
            
            # Output Dir Path
            output_dir = os.path.join(conf.PATHS["base_output_dir"], conf.MODELS["foundational_model"].lower(), subfolder_name.lower())

            
            pipeline: PromptTuningPipeline = PromptTuningPipeline(
                model_name=conf.MODELS["foundational_model"],
                dataset_path=conf.DATA_PATH["dataset_path"],
                output_dir=output_dir,
                local_model_dir=local_model_dir,
                device=conf.DEVICE
            )
            
            if action == actions['train']:
                # Check if the name already exist
                if os.path.isdir(output_dir):
                    print(f"The directory {output_dir} already exists.")
                    continue
                
                pipeline.train(
                    # Change Default Values If Needed. Example:
                    # epochs=5
                    # dataset_processor = Aviman1DatasetProcessor
                    dataset_processor = GooglePersonaDatasetProcessor
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
    4.  [-] Talk with Avi about the dataset - get a better one.
    5.  [-] Complete the other strategies.
    6.  [-] Test the model before training.
    7.  [-] Test the model after small amount of epochs training.
    8.  [-] Ask Amos if we want a lot of epochs - intentional overfiting?
    9.  [v] Find a smaller chat bot model.
    10. [-] Dive into the Prompt Engineering part - Read about it and how to use peft lib. 
""" 
