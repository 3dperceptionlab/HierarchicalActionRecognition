import torch, os
import shutil

def save_epoch(epoch, model, optimizer, work_dir, filename):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                os.path.join(work_dir, filename))

def save_best(working_dir, filename, epoch):
    shutil.copy(os.path.join(working_dir, filename), os.path.join(working_dir, f'best_epoch_{epoch}.pt'))