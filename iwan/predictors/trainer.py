from typing import Dict, Union
import torch
import torch.optim as optimisers
from torch.utils.data import DataLoader, TensorDataset
from .loss import Loss
from .score_predictor.model import WorldCupScorePredictor
from .goal_probability_predictor import WorldCupGoalsPredictor
from copy import deepcopy
import numpy as np

class Trainer:
    def __init__(self, model: Union[WorldCupScorePredictor], config: Dict[str, Dict], tensors: Dict[str, torch.Tensor]):
        self.model = model
        self.best_model = deepcopy(model)
        self.config = config
        self.loss_fn = Loss(config, model.__class__.__name__)
        self.train_loader, self.test_loader = self.construct_dataloaders(tensors)
        self.optimiser = self.make_optimiser()
        self.history = {"train_loss": [], "eval_loss": [], "best_epoch_number": 0}

    
    def train(self, outdir):
        best_eval_loss = np.inf
        best_state_dict = self.model.state_dict()
        for epoch in range(self.config["training"]["epochs"]):
            best_eval_loss, state_dict = self.run_epoch(epoch, best_eval_loss)
            if state_dict is not None:
                self.history["best_epoch_number"] = epoch
                best_state_dict = state_dict
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}      Train Loss: {self.history["train_loss"][-1]}        Eval Loss: {self.history["eval_loss"][-1]}      Best Epoch: {self.history["best_epoch_number"]}')
        
        torch.save(best_state_dict, outdir / "model.pt")
        self.best_model.load_state_dict(best_state_dict)

    
    def run_epoch(self, epoch, best_eval_loss):
        # eval
        self.model.eval()
        eval_batch_losses = []
        state_dict = None
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_batch, y_pred).item()
                eval_batch_losses.append(loss)
                if loss < best_eval_loss:
                    best_eval_loss = loss
                    state_dict = self.model.state_dict()
        # train
        self.model.train()
        train_batch_losses = []
        for x_batch, y_batch in self.train_loader:
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_batch, y_pred)
            train_batch_losses.append(loss.item())
            loss.backward()
            self.optimiser.step()
        
        eval_loss = np.mean(eval_batch_losses)
        train_loss = np.mean(train_batch_losses)
        self.history["train_loss"].append(train_loss)
        self.history["eval_loss"].append(eval_loss)
        
        return best_eval_loss, state_dict


    def make_optimiser(self):
        optim_cfg = deepcopy(self.config["training"]["optimiser"])
        optimiser_name = optim_cfg.pop("optimiser")
        optimiser_class_obj = getattr(optimisers, optimiser_name)
        return optimiser_class_obj(self.model.parameters(), **optim_cfg)
        

    def construct_dataloaders(self, tensors: Dict[str, torch.Tensor]) -> TensorDataset:
        train_dataset = TensorDataset(tensors["x_train"], tensors["y_train"])
        test_dataset = TensorDataset(tensors["x_test"], tensors["y_test"])
        test_size = len(tensors["x_test"])
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["training"]["dataloaders"]["train"]["batch_size"],
            shuffle=self.config["training"]["dataloaders"]["train"]["shuffle"],
            drop_last=self.config["training"]["dataloaders"]["train"]["drop_last"]
        )
        test_batch_size = self.config["training"]["dataloaders"]["eval"]["batch_size"]
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_size if test_batch_size is None else test_batch_size,
            shuffle=self.config["training"]["dataloaders"]["eval"]["shuffle"],
            drop_last=self.config["training"]["dataloaders"]["eval"]["drop_last"]
        )
        return train_loader, test_loader