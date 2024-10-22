import torch
import numpy as np
from tqdm import tqdm
import wandb, math, sys
import torch.nn.functional as F

@torch.no_grad()
def evaluate(epoch, model, dataloader, config, coarse_criterion, fine_criterion):
    num = 0
    corr_1_coarse = 0
    corr_3_coarse = 0
    corr_5_coarse = 0
    corr_1_fine = 0
    corr_3_fine = 0
    corr_5_fine = 0

    fine_1_predictions = []
    anno_ids = []
    video_names = []

    model.eval()
    if not config.model.fine_only:
        coarse_criterion.eval()
    fine_criterion.eval()

    for batch,(rgb_t, flow_t, text, coarse_target, fine_target) in enumerate(tqdm(dataloader)):
        wandb.run.summary["eval_batch"] = batch
        if config.data.rgb_features is not None:
            b, _, _ = rgb_t.shape
        elif config.data.flow_features is not None:
            b, _, _ = flow_t.shape
        else:
            b, _ = text.shape
        num += b

        rgb_t = rgb_t.cuda()
        flow_t = flow_t.cuda()
        text = text.cuda()
        input_data = (rgb_t, flow_t, text)
        coarse_target = coarse_target.cuda()
        fine_target = fine_target.cuda()

        coarse_pred, fine_pred = model(input_data)
        coarse_pred = F.sigmoid(coarse_pred)
        fine_pred = F.sigmoid(fine_pred)

        fine_loss = fine_criterion(fine_pred, fine_target)
        if not config.model.fine_only:
            coarse_loss = coarse_criterion(coarse_pred, coarse_target)
            losses = coarse_loss + fine_loss
        else:
            coarse_loss = torch.tensor(0.0)
            losses = fine_loss
        
        if not math.isfinite(losses):
            print("Loss is infinite during evaluation. Terminating.")
            wandb.alert("Loss is infinite during evaluation. Terminating.")
            sys.exit(1)

        if batch % config.logging.freq == 0:
            wandb.log({"eval_loss": losses.item(), "eval_coarse_loss": coarse_loss.item(), "eval_fine_loss": fine_loss.item()})
            print(f"[{epoch}/{config.solver.epochs}] Eval loss: {losses.item()}")

        if not config.model.fine_only:
            coarse_pred = coarse_pred.cpu()
            coarse_target = coarse_target.cpu()
            _, coarse_1_idx = torch.topk(coarse_pred, 1, dim=-1)
            _, coarse_3_idx = torch.topk(coarse_pred, 3, dim=-1)
            _, coarse_5_idx = torch.topk(coarse_pred, 5, dim=-1)

        _, fine_1_idx = torch.topk(fine_pred, 1, dim=-1)
        _, fine_3_idx = torch.topk(fine_pred, 3, dim=-1)
        _, fine_5_idx = torch.topk(fine_pred, 5, dim=-1)


        for i in range(b):
            if not config.model.fine_only and coarse_target[i][coarse_1_idx[i][0]] == 1:
                corr_1_coarse += 1
            if fine_target[i][fine_1_idx[i][0]] == 1:
                corr_1_fine += 1
            for j in range(3):
                if not config.model.fine_only and coarse_target[i][coarse_3_idx[i][j]] == 1:
                    corr_3_coarse += 1
                    break
            for j in range(3):
                if fine_target[i][fine_3_idx[i][j]] == 1:
                    corr_3_fine += 1
                    break
            for j in range(5):
                if not config.model.fine_only and coarse_target[i][coarse_5_idx[i][j]] == 1:
                    corr_5_coarse += 1
                    break
            for j in range(5):
                if fine_target[i][fine_5_idx[i][j]] == 1:
                    corr_5_fine += 1
                    break
    

    if not config.model.fine_only:
        coarse_acc_1 = corr_1_coarse / num
        coarse_acc_3 = corr_3_coarse / num
        coarse_acc_5 = corr_5_coarse / num
    else:
        coarse_acc_1 = 0.0
        coarse_acc_3 = 0.0
        coarse_acc_5 = 0.0
    fine_acc_1 = corr_1_fine / num
    fine_acc_3 = corr_3_fine / num
    fine_acc_5 = corr_5_fine / num

    wandb.log({"coarse_acc_1": coarse_acc_1, "coarse_acc_5": coarse_acc_5, "fine_acc_1": fine_acc_1, "fine_acc_5": fine_acc_5, "fine_acc_3": fine_acc_3})
    print(f"[{epoch}/{config.solver.epochs}] Eval coarse acc@1: {coarse_acc_1}, coarse acc@5: {coarse_acc_5} \\ fine acc@1: {fine_acc_1}, fine acc@3: {fine_acc_3}, fine acc@5: {fine_acc_5}")

    return fine_acc_1, fine_acc_3, fine_acc_5, coarse_acc_1, coarse_acc_3, coarse_acc_5