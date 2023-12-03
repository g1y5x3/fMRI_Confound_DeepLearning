import os
import yaml
import wandb
from train import train

os.environ["WANDB"] = "1"

if __name__ == "__main__":

  with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  run = wandb.init(config = config)
  
  train(wandb.config, run)
  
