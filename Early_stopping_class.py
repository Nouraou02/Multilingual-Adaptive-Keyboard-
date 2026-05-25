from xml.parsers.expat import model

import torch
import os

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < (self.best_loss - self.min_delta):
            print(f"✅ Val Loss Improved ({self.best_loss:.4f} -> {val_loss:.4f}). Saving model...")
            self.best_loss = val_loss
            self.counter = 0
            # Save the LoRA adapter weights
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(self.save_path)
            else:
                torch.save(model.state_dict(), self.save_path + '.pth')
        else:
            self.counter += 1
            print(f"⚠️ No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True