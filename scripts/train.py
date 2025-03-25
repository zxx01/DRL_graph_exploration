import os
import pickle
import time
import subprocess
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import TrainingConfig
from model_factory import ModelFactory
from policy import DeepQ, A2C

def setup_training():
    """Setup training configuration and create necessary directories."""
    config = TrainingConfig(
        training_method="DQN",  # DQN, A2C
        model_name="NoisyGCN",  # GCN, GG-NN, g-U-Net, DuelingGCN, NoisyGCN
        use_double_dqn=True,
        exploration_method="noisy"  # None, noisy, epsilon, bayesian
    )
    
    # Create necessary directories
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.object_path, exist_ok=True)
    
    return config

def create_and_save_training_object(config: TrainingConfig):
    """Create and save the training object."""
    device = torch.device(config.device)
    
    if config.training_method == "DQN":
        training = DeepQ(config.case_path, config.model_name, device)
    else:  # A2C
        training = A2C(config.case_path, device)
    
    # Save training object
    with open(f"{config.object_path}saved_training.pkl", 'wb') as f:
        pickle.dump(training, f, pickle.HIGHEST_PROTOCOL)
    
    return training

def save_initial_models(config: TrainingConfig):
    """Create and save initial model states."""
    if config.training_method == "DQN":
        policy_model, target_model = ModelFactory.create_dqn_models(config)
        torch.save(policy_model.state_dict(), config.model_paths['policy'])
        torch.save(target_model.state_dict(), config.model_paths['target'])
    else:  # A2C
        policy_model, value_model = ModelFactory.create_a2c_models(config)
        torch.save(policy_model.state_dict(), config.model_paths['policy'])
        torch.save(value_model.state_dict(), config.model_paths['value'])

def run_training_epochs(config: TrainingConfig, training):
    """Run training epochs and log metrics."""
    writer = SummaryWriter(log_dir=config.log_path)
    time_total = 0
    
    # Calculate number of epochs
    epoch_nums = training.EXPLORE / training.epoch
    
    for i in range(int(epoch_nums)):
        # Prepare command
        print(f"Running training epoch {i+1} of {int(epoch_nums)}")
        cmd = f"python3 run_training.py {config.training_method} {config.model_name} {str(config.use_double_dqn).lower()}"
        if config.exploration_method:
            cmd += f" {config.exploration_method}"
        
        # Run training epoch
        time_start = time.time()
        subprocess.call(cmd, shell=True)
        duration = time.time() - time_start
        time_total += duration
        print(f"10000 epoches time: {duration} s.")
        
        # Log metrics
        log_metrics(config, writer)
    
    print(f"1e6 total time: {time_total} s.")

def log_metrics(config: TrainingConfig, writer: SummaryWriter):
    """Log training metrics to tensorboard."""
    # Load and log reward data
    reward_data = np.loadtxt(f"{config.object_path}temp_reward.csv", delimiter=",")
    for step, reward in reward_data:
        writer.add_scalar('Train/avg_reward', reward, step)
    
    # Load and log loss data
    loss_data = np.loadtxt(f"{config.object_path}temp_loss.csv", delimiter=",")
    for step, loss in loss_data:
        writer.add_scalar('Train/loss', loss, step)

def main():
    # Setup training configuration
    config = setup_training()
    
    # Create and save training object
    training = create_and_save_training_object(config)
    
    # Save initial models
    save_initial_models(config)
    
    # Run training epochs
    run_training_epochs(config, training)

if __name__ == "__main__":
    main()
