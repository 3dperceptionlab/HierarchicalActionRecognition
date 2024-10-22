import os, time
import yaml
import shutil
import argparse
import math, sys
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap
from pprint import PrettyPrinter

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.tsu import ToyotaSmartHomeDataset
from modules.aggregation import AggregationTransformer
from utils.solver import _optimizer, _lr_scheduler
from utils.saving import save_epoch, save_best
from utils.evaluation import evaluate
import numpy as np
import random

def get_config():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--log_time", type=str, default=None, help="Current time for logging purposes")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
  
    config['working_dir'] = os.path.join("./exp", config['name'], args.log_time)
    # Log config
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(config['working_dir']))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)
    
    config = DotMap(config)

    # Set the working directory
    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config.working_dir)
    shutil.copy("main.py", config.working_dir)

    # Set wandb
    wandb.require('core')
    wandb.init(project="NEWActionHierarchies",
                name="{}_{}".format(config.name, args.log_time),
                config=config)

    return config


def main():
    config = get_config()

    # Set the seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("===== WARNING =====")
        print("Running on CPU")
        print("==================")
        wandb.alert("Running on CPU")
        import sys; sys.exit(1)

    # Get dataset
    train_ds  = ToyotaSmartHomeDataset(config, split="train")
    test_ds = ToyotaSmartHomeDataset(config, split="test")

    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False)


    # Create the model
    model = AggregationTransformer(config)
    model = model.cuda()

    # Create the optimizer and lr_scheduler
    optimizer = _optimizer(config, model)
    lr_scheduler = _lr_scheduler(config, optimizer)


    fine_criterion = nn.BCEWithLogitsLoss()
    coarse_criterion = None
    if not config.model.fine_only:
        coarse_criterion = nn.BCEWithLogitsLoss()

    if config.eval:
        # Load the model weights
        model.load_state_dict(torch.load(config.load)['model_state_dict'])
        evaluate(0, model, test_loader, config, coarse_criterion, fine_criterion)
        wandb.finish()
        return


    print(model)
    best = (0.0,0.0,0.0,0.0,0.0,0.0)
    early_stopping = 1
    
    
    # Train the model
    for epoch in range(config.solver.epochs):
        wandb.run.summary["epoch"] = epoch
        model.train()
        fine_criterion.train()
        if not config.model.fine_only:
            coarse_criterion.train()

        for batch,(rgb_t, flow_t, text, coarse_target, fine_target) in enumerate(tqdm(train_loader)):
            wandb.run.summary["batch"] = batch
            if (batch+1) == 0 or (batch+1) % 10 == 0:
                lr_scheduler.step(epoch + batch / len(train_loader))
            optimizer.zero_grad()

            rgb_t = rgb_t.cuda()
            flow_t = flow_t.cuda()
            text = text.cuda()
            input_data = (rgb_t, flow_t, text)
            coarse_target = coarse_target.cuda()
            fine_target = fine_target.cuda()

            coarse_pred, fine_pred = model(input_data)

            # Compute loss
            fine_loss = fine_criterion(fine_pred, fine_target)
            if not config.model.fine_only:
                coarse_loss = coarse_criterion(coarse_pred, coarse_target)
                losses = coarse_loss + fine_loss
            else:
                coarse_loss = torch.tensor(0.0)
                losses = fine_loss

            if not math.isfinite(losses):
                print("Loss is infinite")
                wandb.alert("Loss is infinite")
                sys.exit(1)

            if batch % config.logging.freq == 0:
                wandb.log({"loss": losses.item(), "coarse_loss": coarse_loss.item(), "fine_loss": fine_loss.item(), "lr": optimizer.param_groups[0]['lr']})
                print(f"[{epoch}/{config.solver.epochs}] Loss: {losses.item()}")

            
            losses.backward()
            if config.solver.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.solver.clip_grad_norm)
            optimizer.step()
        
        if epoch % config.solver.eval_freq == 0:
            print(f"[{epoch}/{config.solver.epochs}] Saving epoch...")
            save_epoch(epoch, model, optimizer, config.working_dir, "last_epoch.pt")
            res = evaluate(epoch, model, test_loader, config, coarse_criterion, fine_criterion)

            if res[0] > best[0]:
                early_stopping = 1
                save_best(config.working_dir, "last_epoch.pt", epoch) # Copy the last_epoch.pt to model_best.pt for faster execution
                print("----- NEW BEST -----")
                print(f"Improvement: {res[0] - best[0]}")
                best = res
            else:
                early_stopping += 1
                if early_stopping == config.solver.early_stopping:
                    print("Early stopping...")
                    break
            wandb.log({"best_fine": best[0], "best_fine_3": best[1], "best_fine_5": best[2], "best_coarse": best[3], "best_coarse_3": best[4], "best_coarse_5": best[5]})
    wandb.finish()

if __name__ == "__main__":
    main()