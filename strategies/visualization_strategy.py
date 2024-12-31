import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy


# Visualization Strategy
class VisualizationStrategy(BasePipelineStrategy):
    """
    A strategy for visualizing training and evaluation logs.
    Extracts loss data from checkpoint logs and generates a loss curve.

    Attributes:
        output_dir (str): Directory where checkpoint logs are saved.
    """
    
    def __init__(self, output_dir: str):
        """
            Initialize the VisualizationStrategy with the output directory.

            Args:
                output_dir (str): Path to the directory containing model checkpoints.
        """
        
        self.output_dir = output_dir

    def get_last_checkpoint_dir(self):
        """
            Identify the latest checkpoint directory.

            Returns:
                str: Path to the most recent checkpoint directory, or None if no checkpoints are found.
        """
        
        checkpoints = [
            os.path.join(self.output_dir, d)
            for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith("checkpoint")
        ]
        if not checkpoints:
            return None
        
        # Sort checkpoints by creation time and return the latest one
        return max(checkpoints, key=os.path.getmtime)

    def execute(self):
        """
            Visualize training and evaluation loss curves based on log data.
        """
        
        print("[INFO] Visualizing training and evaluation logs.")

        last_checkpoint_dir = self.get_last_checkpoint_dir()
        if not last_checkpoint_dir:
            print("[ERROR] No checkpoint directories found.")
            return

        log_file = os.path.join(last_checkpoint_dir, "trainer_state.json")
        if not os.path.exists(log_file):
            print(f"[ERROR] Log file not found: {log_file}")
            return

        with open(log_file, "r") as f:
            log_data = json.load(f)

        if "log_history" not in log_data:
            print("[ERROR] Log history not found in training logs.")
            return

        log_history = log_data["log_history"]
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_losses = []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])
            if "eval_loss" in entry and "step" in entry:
                eval_steps.append(entry["step"])
                eval_losses.append(entry["eval_loss"])

        if not train_steps or not train_losses:
            print("[ERROR] No training loss data found in logs.")
            return
        if not eval_steps or not eval_losses:
            print("[WARNING] No evaluation loss data found in logs.")

        # Plot training and evaluation loss
        plt.figure(figsize=(10, 6))
        if train_steps and train_losses:
            plt.plot(train_steps, train_losses, label="Training Loss", color="blue")
        if eval_steps and eval_losses:
            plt.plot(eval_steps, eval_losses, label="Evaluation Loss", color="orange")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
