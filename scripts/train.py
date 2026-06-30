import os
import sys
import torch
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solvers.notebook_transformer import TSP_Transformer
from app.solvers.dataset import load_pkl_dataset
from app.training.trainer import TSPTrainer

def load_datasets():
    # Load dataset from local pkl files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "misc", "data")
    
    train_pkl = os.path.join(data_dir, "train_20_DLL_ass4.pkl")
    valid_pkl = os.path.join(data_dir, "valid_20_DLL_ass4.pkl")
    test_pkl = os.path.join(data_dir, "test_20_DLL_ass4.pkl")
    
    if os.path.exists(valid_pkl) and os.path.exists(test_pkl):
        print("Loading datasets from misc/data/...")
        try:
            if os.path.exists(train_pkl) and os.path.getsize(train_pkl) > 0:
                train_x, train_y = load_pkl_dataset(train_pkl)
            else:
                train_x, train_y = load_pkl_dataset(valid_pkl)
            val_x, val_y = load_pkl_dataset(test_pkl)
            
            # Slice to validation sample sizes
            train_x_t = torch.tensor(train_x[:2000], dtype=torch.float32)
            train_y_t = torch.tensor(train_y[:2000], dtype=torch.long)
            val_x_t = torch.tensor(val_x[:200], dtype=torch.float32)
            val_y_t = torch.tensor(val_y[:200], dtype=torch.long)
            
            print(f"Loaded train shape: {train_x_t.shape}, val shape: {val_x_t.shape}")
            return train_x_t, train_y_t, val_x_t, val_y_t
        except Exception as e:
            print(f"Error loading pkl files: {e}")
            sys.exit(1)
    else:
        print("Error: misc/data/ pkl files not found. Generate them before running training.")
        sys.exit(1)

def run_training_pipeline():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load datasets
    train_x, train_y, val_x, val_y = load_datasets()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # Training Configurations
    epochs = 12
    batch_size = 64
    learning_rate = 1e-3
    
    # 1. Train Standard Notebook Transformer
    standard_model = TSP_Transformer(
        num_cities=20, 
        d_model_enc=32, 
        d_model_dec=32, 
        d_model_ff=64, 
        nhead=8, 
        num_layers_enc=3, 
        num_layers_dec=3, 
        dropout_rate=0.3
    )
    
    std_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_gradient_accumulation": False
    }
    
    std_trainer = TSPTrainer(
        model=standard_model,
        config=std_config,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        device=device
    )
    std_trainer.train(save_filename="model_transformer.pth")
    
    # 2. Train Gradient Accumulation Notebook Transformer
    ga_model = TSP_Transformer(
        num_cities=20, 
        d_model_enc=32, 
        d_model_dec=32, 
        d_model_ff=64, 
        nhead=8, 
        num_layers_enc=3, 
        num_layers_dec=3, 
        dropout_rate=0.3
    )
    
    ga_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_gradient_accumulation": True,
        "accumulation_steps": 4
    }
    
    ga_trainer = TSPTrainer(
        model=ga_model,
        config=ga_config,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        device=device
    )
    ga_trainer.train(save_filename="model_transformer_GA.pth")

if __name__ == "__main__":
    run_training_pipeline()
