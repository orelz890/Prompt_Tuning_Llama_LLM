import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy

import json
import matplotlib.pyplot as plt


# Visualization Strategy
class VisualizationStrategy(BasePipelineStrategy):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def execute(self):
        print("[INFO] Visualizing training logs.")
        log_file = os.path.join(self.log_dir, "trainer_state.json")

        if not os.path.exists(log_file):
            print(f"[ERROR] Log file not found: {log_file}")
            return

        with open(log_file, "r") as f:
            log_data = json.load(f)

        if "log_history" not in log_data:
            print("[ERROR] Log history not found in training logs.")
            return

        log_history = log_data["log_history"]
        steps = []
        losses = []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])

        if not steps or not losses:
            print("[ERROR] No loss data found in logs.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
