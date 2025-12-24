#!/usr/bin/env python
# coding: utf-8

import os
import json
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Config:
    def __init__(self):
        self.setup_vars()
        self.setup_paths()

    def setup_vars(self):
        ## Reproducibility
        self.np_seed = 42
        self.torch_seed = 36

        ## Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4

        ## Hyperparameters
        self.test_ratio = 0.15
        self.k_folds = 7
        self.num_epochs = 10
        self.batch_size = 2
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001 # L2 regularization
        self.patience = 10 # early stopping patience
        
    def setup_paths(self):
        self.subj_list_path = os.path.join("..", "data", "metadata", "subj_list.txt")
        self.subj_infos_path = os.path.join("..", "data", "metadata", "subj_infos.csv")
        self.img_path_template = os.path.join("..", "data", "processed", "preproc_MNI_wholebrain_cropped", 
                                              "{}_ses-01_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz")
        prefix = datetime.today().strftime('%Y-%m-%d')
        self.output_dir = os.path.join("..", "outputs", f"{prefix}_sfcn")
        os.makedirs(self.output_dir, exist_ok=True)
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.log_path = os.path.join(self.output_dir, "log.txt")
        self.mdl_path_template = os.path.join(self.output_dir, "model_fold-{}.pt")
        self.trainval_hist_path = os.path.join(self.output_dir, "train-val_history.csv")
        self.test_preds_path = os.path.join(self.output_dir, "test_predictions.csv")
        self.test_res_path = os.path.join(self.output_dir, "test_results.json")

class MyData(Dataset):
    def __init__(self, config):
        super(MyData, self).__init__()
        subj_list, subj_infos = self.load_labels(config)
        self.samples = self.make_dataset(config, subj_list, subj_infos)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, *_ = self.samples[idx]
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return img_tensor, label_tensor

    def load_labels(self, config):
        subj_infos = pd.read_csv(config.subj_infos_path)
        subj_infos.set_index("SID", inplace=True)
        # subj_infos["Sex"] = subj_infos["Sex"].map({"M": 0, "F": 1}).astype(int)
        subj_list = pd.read_csv(config.subj_list_path, header=None)[0].values
        return subj_list, subj_infos.loc[subj_list, :]

    def make_dataset(self, config, subj_list, subj_infos):
        samples = []
        for sid in subj_list:
            img_path = config.img_path_template.format(sid)
            age = subj_infos.loc[sid, "Age"]
            age_group = "Y" if age < 45 else "O"
            sex = subj_infos.loc[sid, "Sex"]
            samples.append((img_path, age, f"{age_group}{sex}", sid))
        return samples

class SFCN(nn.Module):
    def __init__(self, c_nums=[1, 32, 64, 128, 256, 256, 64, 1], add_dropout=True): 
        super(SFCN, self).__init__()
        self.feature_extractor = nn.Sequential()

        for i, (c_in, c_out) in enumerate(zip(c_nums[:-1], c_nums[1::])):
            if i < (len(c_nums) - 2): 
                add_pool = (i <= 5)
                k, p = (3, 1) if add_pool else (1, 0)
                block = self.build_block(i, c_in, c_out, add_pool, kernel_size=k, padding=p)
                self.feature_extractor.add_module(f"block_{i}", block)
            else:
                self.regressor = nn.Sequential(OrderedDict([
                    ("avg_pool", nn.AdaptiveAvgPool3d(1))
                ]))
                if add_dropout:
                    self.regressor.add_module("dropout", nn.Dropout(0.5))
                self.regressor.add_module(f"conv_{i}", nn.Conv3d(c_in, c_out, 1))

    @staticmethod
    def build_block(i, c_in, c_out, add_pool, kernel_size=3, padding=0):
        block = nn.Sequential(OrderedDict([
            (f"conv_{i}", nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding)), 
            (f"batch_norm_{i}", nn.BatchNorm3d(num_features=c_out))
        ]))
        if add_pool:
            block.add_module(f"max_pool_{i}", nn.MaxPool3d(kernel_size=2, stride=2))
        block.add_module(f"relu_{i}", nn.ReLU())
        return block

    def forward(self, x):
        # print("Input shape:", x.shape)
        x_f = self.feature_extractor(x)
        # print("After feature extraction:", x_f.shape)
        out = self.regressor(x_f)
        # print("After regression:", out.shape)
        return out.view(out.size(0), -1) # flattening

def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert np.ndarray to list
    elif isinstance(obj, np.generic): 
        return obj.item()   # Convert np.generic to scalar
    elif isinstance(obj, pd.Series):
        return obj.tolist() # Convert pd.Series to list
    elif isinstance(obj, list):
        return [ convert_np_types(i) for i in obj ] 
    elif isinstance(obj, dict):
        return { k: convert_np_types(v) for k, v in obj.items() } 
    else:
        return obj

def save_config(config):
    config_dict = {
        k.upper(): v for k, v in config.__dict__.items() 
        if not any(s in k for s in ["path", "dir"])
    }
    config_dict = convert_np_types(config_dict)
    with open(config.config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False)

def setup_logger(config, logger_name="sfcn_logger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        handler = logging.FileHandler(config.log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def train_or_eval(mode, loader, model, optimizer, criterion, scaler, config):
    avg_loss = 0.0
    y_true, y_pred = [], []

    if mode == "train":
        model.train()
        for batch_n, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True) # clear gradients
            with autocast(device_type=config.device.type, dtype=torch.float16):
                outputs = model(imgs) # forward propagation
                loss = criterion(outputs, labels) # compute loss
            scaler.scale(loss).backward() # compute gradients
            scaler.step(optimizer) # update weights
            scaler.update()
            avg_loss += loss.item() * imgs.size(0) # weighted by batch size
            y_true.append(labels.detach().cpu().numpy().reshape(-1))
            y_pred.append(outputs.detach().cpu().numpy().reshape(-1))

    elif mode == "eval":
        model.eval()
        with torch.no_grad():
            for batch_n, (imgs, labels) in enumerate(loader):
                imgs = imgs.to(config.device)
                labels = labels.to(config.device)
                with autocast(device_type=config.device.type, dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                avg_loss += loss.item() * imgs.size(0)
                y_true.append(labels.detach().cpu().numpy().reshape(-1))
                y_pred.append(outputs.detach().cpu().numpy().reshape(-1))

    avg_loss /= len(loader.dataset) # weighted average
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return avg_loss, {"MAE": mae, "RMSE": rmse, "R2": r2}, y_true, y_pred

def main():
    ## Setups
    config = Config()
    save_config(config)
    
    np.random.seed(config.np_seed)
    torch.manual_seed(config.torch_seed)

    logger = setup_logger(config)
    trainval_history = []

    ## Prepare dataset 
    dataset = MyData(config)
    data_idxs, data_grps = (
        np.array([ [idx, grp] for idx, (_, _, grp, _) in enumerate(dataset.samples) ]).T
    )

    ## Split test set
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.test_ratio, random_state=config.np_seed
    )
    trainval_idx, test_idx = next(sss.split(data_idxs, data_grps))
    loader_test = DataLoader(
        Subset(dataset, test_idx), 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=(config.device.type == "cuda")
    )

    ## K-fold cross-validation
    skf = StratifiedKFold(
        n_splits=config.k_folds, shuffle=True, random_state=config.np_seed
    )
    for fold_n, (train_idx, val_idx) in enumerate(skf.split(trainval_idx, data_grps[trainval_idx])):
        log_text = f"Fold [{fold_n + 1}/{config.k_folds}]"

        loader_train = DataLoader(
            Subset(dataset, train_idx), 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            pin_memory=(config.device.type == "cuda")
        )
        loader_val = DataLoader(
            Subset(dataset, val_idx), 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=(config.device.type == "cuda")
        )

        model = SFCN().to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss().to(config.device)
        scaler = GradScaler()

        best_val = np.inf
        best_state = None
        bad_epochs = 0

        ## Training and validation loop
        for epoch_n in range(config.num_epochs):
            log_text += f", Epoch [{epoch_n + 1}/{config.num_epochs}]"

            train_loss, *_ = train_or_eval(
                "train", loader_train, model, optimizer, criterion, scaler, config
            )
            val_loss, val_metrics, *_ = train_or_eval(
                "eval", loader_val, model, optimizer, criterion, scaler, config
            )
            trainval_history.append({
                "fold": fold_n + 1,
                "epoch": epoch_n + 1,
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            ## Early stopping
            if val_metrics["RMSE"] < best_val:
                best_val = val_metrics["RMSE"]
                best_epoch = epoch_n + 1
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.patience:
                    break

            ## Restore best model state
            if best_state is not None:
                model.load_state_dict(best_state)

            log_text += f", Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_metrics['RMSE']:.4f}"
            logger.info(log_text)

        ## Save best model of the fold
        torch.save({
            "fold": fold_n + 1, 
            "best_epoch": best_epoch,
            "model_state_dict": best_state
        }, config.mdl_path_template.format(fold_n + 1))

    ## Save training and validation history
    trainval_hist_df = pd.DataFrame(trainval_history)
    trainval_hist_df.to_csv(config.trainval_hist_path, index=False)

    ## Evaluate on test set using the best model from all folds
    best_fold = trainval_hist_df.sort_values(by=["val_RMSE"], ascending=True).iloc[0]["fold"]
    best_model_checkpoint = torch.load(config.mdl_path_template.format(best_fold), map_location=config.device)
    best_model = SFCN().to(config.device)
    best_model.load_state_dict(best_model_checkpoint["model_state_dict"])
    test_loss, test_metrics, y_true_test, y_pred_test = train_or_eval(
        "eval", loader_test, best_model, optimizer, criterion, scaler, config
    )
    logger.info(f"Best Fold: {best_fold}, Test Loss: {test_loss:.4f}, Test MAE: {test_metrics['MAE']:.4f}, Test RMSE: {test_metrics['RMSE']:.4f}, Test R2: {test_metrics['R2']:.4f}")

    pd.DataFrame({
        "SID": [dataset.samples[idx][-1] for idx in test_idx],
        "True_Age": y_true_test,
        "Predicted_Age": y_pred_test
    }).to_csv(config.test_preds_path, index=False)

    results = convert_np_types({
        "best_fold": best_fold, 
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()}
    })
    with open(config.test_res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
    print("Done :-)")

