# Standard Imports
import torch, math
import numpy as np

from torch import nn
from transformers import ViTModel
from torch.utils.data import random_split, Subset

# Custom Imports
from train_class import Trainer
from factenc_vivit import ViVit
from train_utils import load_vit_weights
from dataset_class import Custom_Traffic_Dataset

class AlternatingWarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps_per_cycle, warmup_fraction=0.1, 
                 min_lr=0, warmup_start_factor=0.1, last_epoch=-1):
        """
        Args:
            optimizer: The optimizer to modify learning rate for
            steps_per_cycle: Number of steps in one complete cycle (warmup + cosine)
            warmup_fraction: Fraction of steps_per_cycle to spend on warmup
            min_lr: Minimum learning rate during cosine phase
            warmup_start_factor: Starting lr factor for warmup (0.1 means start at 0.1 * base_lr)
        """
        self.steps_per_cycle = steps_per_cycle
        self.warmup_steps = int(steps_per_cycle * warmup_fraction)
        self.cosine_steps = steps_per_cycle - self.warmup_steps
        self.min_lr = min_lr
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate which cycle we're in and the step within that cycle
        current_cycle = self._step_count // self.steps_per_cycle
        step_in_cycle = self._step_count % self.steps_per_cycle
        
        # Check if we're in warmup or cosine phase of the current cycle
        if step_in_cycle < self.warmup_steps:
            # Warmup phase
            warmup_progress = step_in_cycle / self.warmup_steps
            # Linear warmup from warmup_start_factor to 1
            factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * warmup_progress
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine phase
            cosine_progress = (step_in_cycle - self.warmup_steps) / self.cosine_steps
            return [
                self.min_lr + (base_lr - self.min_lr) * 
                (1 + math.cos(math.pi * cosine_progress)) / 2
                for base_lr in self.base_lrs
            ]
            
class WarmupLinearDecayLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer: Optimizer to modify learning rate for
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate after decay (default: 0.0)
            last_epoch: The index of last epoch (default: -1)
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lrs_orig = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs_orig]
        else:
            # Linear decay
            alpha = 1 - (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            alpha = max(0.0, alpha)
            return [self.min_lr + (base_lr - self.min_lr) * alpha 
                   for base_lr in self.base_lrs_orig]

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEED = 3000
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(SEED)
    
    # Dataset Parameters
    image_size = 384              # Height and width of frame
    tube_hw = 32            # height and width of tubelet, frame size must be divisible by this number
    latent_size = 1024       # Size of embedding,
    batch_size = 80        # batch size
    
    subst = False
    subst_ratio = 0.001
    #subst_ratio = 0.00011
    
    # Model Parameters 
    num_class = 2           # Num of classes
    num_heads = 16          # Number of attention heads in a single encoder
    num_spatial_encoders = 24       # Number of encoders in model
    
    #Training Parameters     
    epochs = 12             # Number of iterations through entirety of Data
    max_lr = 3.33e-5          # learning rate
    min_lr = 1e-7
    accumulated_steps = 4 # number of forward passes before updating weights (Effective batch size = batch_size * accumulated_steps)
    #TODO: Reapply regularization
    weight_decay = 0 # 1e-4        # Weight Decay
    dropout = 0.2           # Dropout
    val_steps = 1000

    #File Management Parameters
    model_num = 1
    load_from_checkpoint = False
    load_checkpoint_path = f'./model28/checkpoint.pth'
    out_checkpoint_path = f'./results/models/model{model_num}/checkpoint.pth'
    best_checkpoint_path = f'./results/models/model{model_num}/best_checkpoint.pth'
    epoch_checkpoint_path = f'./results/models/model{model_num}/first_epoch_checkpoint.pth'
    
    ###Testing Params
    interval = 8
    tube_d = 8             # depth of tubelet, i.e. number of frames back 
    num_temporal_encoders = 4       # Number of encoders in model
    n_channels = 1          # R,G,B -> Gray
    
    if n_channels == 3:
        mean= [0.41433686, 0.41618344, 0.4186155 ]  # Calculated from dataset
        std= [0.1953987,  0.19625916, 0.19745596]   # Calculated from dataset
    else:
        mean = [0.41735464]
        std = [0.1976403]
    
    # Models
    print('Loading Model...')
    #--depth = 2
    model = ViVit(num_spatial_encoders, num_temporal_encoders, latent_size, device, num_heads, num_class, dropout, tube_hw, tube_d, n_channels, batch_size, image_size)
    vit = ViTModel.from_pretrained('google/vit-large-patch32-384')

    # Load ViT weights into spatial encoder
    model_dict = load_vit_weights(model.state_dict(), vit.state_dict(), latent_size, SEED)
    model.load_state_dict(model_dict, strict=True)
    del vit, model_dict
    print('Loaded weights successfully')

    print('Loading Dataset...')
    train_data = Custom_Traffic_Dataset(tube_d, image_size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt', mean, std, interval)
    train_data, test_data, val_data = random_split(train_data, [0.7, 0.2, 0.1], generator=generator)
    datasets = {'train': train_data, 'test': test_data, 'val': val_data}
    
    if subst:
        # Calculate the number of samples for the smaller training set (e.g., 10% of train_data)
        small_train_size = int(subst_ratio * len(train_data))

        # Create a subset of the train_data
        small_train_data = Subset(train_data, torch.arange(small_train_size))

        # Replace the original train_data in the datasets dictionary
        datasets['train'] = small_train_data
    
    print('Initializing Dataloader...')
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers = False,
        drop_last=True,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
        ) for x in ['train','test', 'val']}

    print('Setting up training parameters...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    #optimizer = ADOPT(model.parameters(), lr = max_lr, decoupled = True)
    
    weights = torch.tensor([1/0.47, 1/0.53])
    class_weights = weights / weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    iters = (len(dataloaders['train']) // accumulated_steps + (1 if len(dataloaders['train']) % accumulated_steps != 0 else 0)) * epochs
    
    # num_warmup_steps = int(round(iters * .05))
    # print(f"Number of warmup steps: {num_warmup_steps}")
    
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor = 0.1,
    #     end_factor = 1.0, 
    #     total_iters = num_warmup_steps
    # )
    # # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    # #     optimizer,
    # #     T_max = iters * epochs,
    # #     eta_min = min_lr
    # # )
    # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0= iters,
    #     T_mult = 5,
    #     eta_min = min_lr
    # )
    
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, main_scheduler],
    #     milestones=[num_warmup_steps]
    # )
    
    # scheduler = AlternatingWarmupCosineScheduler(
    #     optimizer,
    #     steps_per_cycle=iters,
    #     warmup_fraction = 0.1,
    #     min_lr=min_lr,
    #     warmup_start_factor=0.1
    # )
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters = iters)
    scheduler = WarmupLinearDecayLR(optimizer, total_steps = iters, warmup_steps = iters//10, min_lr=min_lr)
    
    
    
    # Create Trainer instance
    trainer = Trainer(model, model_num, dataloaders, criterion, optimizer, scheduler, device, out_checkpoint_path, load_checkpoint_path, best_checkpoint_path, epoch_checkpoint_path, load_from_checkpoint, accumulated_steps, iters)
    
    # Train the model
    metrics = trainer.train(epochs=epochs, batch_size=batch_size, val_steps=val_steps)
    

if __name__ == '__main__':
    main()