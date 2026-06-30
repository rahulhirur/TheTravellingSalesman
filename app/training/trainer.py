import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

class TSPTrainer:
    def __init__(self, model, config, train_x, train_y, val_x, val_y, device=None):
        self.model = model
        self.config = config
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def train(self, save_filename="model_transformer.pth"):
        epochs = self.config.get("epochs", 10)
        batch_size = self.config.get("batch_size", 64)
        learning_rate = self.config.get("learning_rate", 1e-3)
        use_ga = self.config.get("use_gradient_accumulation", False)
        accumulation_steps = self.config.get("accumulation_steps", 4)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        
        # Setup local save paths
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(parent_dir, "misc")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_filename)
        
        n_train = self.train_x.shape[0]
        n_val = self.val_x.shape[0]
        num_nodes = self.train_x.shape[1]
        
        # Initialize Weights & Biases Run
        run_name = f"transformer_{save_filename.split('.')[0]}_{int(time.time())}"
        wandb.init(
            project="TheTravellingSalesman",
            name=run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_gradient_accumulation": use_ga,
                "accumulation_steps": accumulation_steps if use_ga else 1,
                "num_nodes": num_nodes,
                "device": str(self.device)
            }
        )
        
        print(f"\nStarting training for {save_filename} (GA={use_ga})...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            # Shuffle dataset
            indices = torch.randperm(n_train)
            shuffled_x = self.train_x[indices]
            shuffled_y = self.train_y[indices]
            
            num_batches = math.ceil(n_train / batch_size)
            
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_train)
                
                x_batch = shuffled_x[start_idx:end_idx].to(self.device)
                y_batch = shuffled_y[start_idx:end_idx].to(self.device)
                
                y_input = y_batch[:, :-1]
                y_target = y_batch[:, 1:].reshape(-1)
                
                # Forward pass
                outputs = self.model(x_batch, y_input)
                outputs = outputs.reshape(-1, num_nodes)
                
                loss = nn.functional.nll_loss(outputs, y_target)
                
                if use_ga:
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (b + 1) % accumulation_steps == 0 or (b + 1) == num_batches:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                train_loss += loss.item() * (end_idx - start_idx) * (accumulation_steps if use_ga else 1)
                
            train_loss = train_loss / n_train
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                num_val_batches = math.ceil(n_val / batch_size)
                for b in range(num_val_batches):
                    start_idx = b * batch_size
                    end_idx = min(start_idx + batch_size, n_val)
                    x_batch = self.val_x[start_idx:end_idx].to(self.device)
                    y_batch = self.val_y[start_idx:end_idx].to(self.device)
                    
                    y_input = y_batch[:, :-1]
                    y_target = y_batch[:, 1:].reshape(-1)
                    
                    outputs = self.model(x_batch, y_input)
                    outputs = outputs.reshape(-1, num_nodes)
                    
                    loss = nn.functional.nll_loss(outputs, y_target)
                    val_loss += loss.item() * (end_idx - start_idx)
                    
            val_loss = val_loss / n_val
            scheduler.step()
            
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
            
            # Log epoch metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr
            })
            
            # Save best checkpoint locally
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  New best model saved locally to {save_path}")
                
        # Register best model weights as a W&B Artifact
        try:
            artifact_name = f"{save_filename.split('.')[0]}_model"
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f"  Successfully registered model artifact in W&B: {artifact_name}")
        except Exception as e:
            print(f"Failed logging model artifact to W&B: {e}")
            
        wandb.finish()
        print(f"Training for {save_filename} completed successfully!")
        return best_val_loss
