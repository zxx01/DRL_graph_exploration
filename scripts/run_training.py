import pickle
import sys
import torch
from config import TrainingConfig
from model_factory import ModelFactory

def parse_args():
    """Parse command line arguments and create configuration."""
    if len(sys.argv) < 3:
        raise ValueError("Usage: python run_training.py <training_method> <model_name> [use_double_dqn] [exploration_method]")
    
    training_method = sys.argv[1]
    model_name = sys.argv[2]
    use_double_dqn = True
    exploration_method = None
    
    if len(sys.argv) > 3:
        use_double_dqn = sys.argv[3].lower() == 'true'
    
    if len(sys.argv) > 4:
        if sys.argv[4] in ["noisy", "epsilon", "bayesian"]:
            exploration_method = sys.argv[4]
    
    return TrainingConfig(
        training_method=training_method,
        model_name=model_name,
        use_double_dqn=use_double_dqn,
        exploration_method=exploration_method
    )

def main():
    # Parse arguments and create configuration
    config = parse_args()
    
    # Load training object
    with open(f"{config.object_path}saved_training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    # Load models
    models = ModelFactory.load_models(config)
    
    # Print configuration
    print(f"Continuing training {config.training_method} with {config.model_name}")
    if config.training_method == "DQN":
        print(f"Using Double DQN: {config.use_double_dqn}")
        print(f"Exploration method: {config.exploration_method or 'auto'}")
    
    # Run training
    if config.training_method == "DQN":
        policy_model, target_model = models
        training.running(
            policy_model, 
            target_model, 
            double_dqn=config.use_double_dqn, 
            exploration_method=config.exploration_method
        )
    else:  # A2C
        policy_model, value_model = models
        training.running(policy_model, value_model)
    
    # Save training state
    with open(f"{config.object_path}saved_training.pkl", 'wb') as f:
        pickle.dump(training, f)

if __name__ == "__main__":
    main()
