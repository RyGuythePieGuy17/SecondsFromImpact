import signal, sys, torch, math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pathlib import Path

from tqdm import tqdm
from train_utils import evaluate_model, print_metrics, save_checkpoint, load_checkpoint, move_to_cpu


class Trainer:
    def __init__(self, model, model_num, dataloaders, criterion, optimizer, scheduler, device, out_checkpoint_path, load_checkpoint_path, best_checkpoint_path, epoch_checkpoint_path, load_from_checkpoint=False, accumulated_steps = 1, iters = 0):
        self.model = model
        self.model_num = model_num
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_checkpoint_path = out_checkpoint_path
        self.load_checkpoint_path = load_checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.epoch_checkpoint_path = epoch_checkpoint_path
        self.load_from_checkpoint = load_from_checkpoint
        self.accumulated_steps = accumulated_steps
        self.iters = iters
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.best_val_loss = 100.0
        
        # This is to ensure plt.show() is called only once
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 16))

        # Initialize metrics
        self.metrics = {
            'train_losses': [],
            'train_lr': [],
            'pred_confidences' : [],
            'val_losses': [],
            'val_accs': [],
            'val_precisions': [],
            'val_recalls': []
        }

        # Load from checkpoint if required
        if self.load_from_checkpoint:
            print("Loading from checkpoint...")
            self.current_epoch, self.metrics, self.current_batch_idx, self.model = load_checkpoint(
                self.model, self.optimizer, self.scheduler, self.load_checkpoint_path, self.device
            )

        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        if torch.utils.data.get_worker_info() is None:
            print('You pressed Ctrl+C! Saving checkpoint...')
        
            # Wait for all CUDA operations to finish
            torch.cuda.synchronize()
        
            save_checkpoint({
                'epoch': self.current_epoch,
                'model_state_dict': move_to_cpu(self.model.state_dict()),
                'optimizer_state_dict':move_to_cpu(self.optimizer.state_dict()),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'batch_idx': self.current_batch_idx,
                'metrics': self.metrics
            }, self.out_checkpoint_path)
            sys.exit(0)
        
    def update_plots(self, metrics):
        # Clear the axes to avoid overlapping plots
        
        path_name = self.out_checkpoint_path.split('/')
        if 'step2' in path_name[2]:
            path_name = f'./results/plots/plot_{self.model_num}_step2.png'
        else:
            path_name = f'./results/plots/plot_{self.model_num}.png'
        for row in self.ax:
            for a in row:
                a.clear()
                
        Path('/'.join(path_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        
        # Training Loss
        if metrics['train_losses']:
            steps, losses = zip(*metrics['train_losses'])
            self.ax[0,0].plot(steps, losses, label='Training Loss')
            self.ax[0,0].set_title('Training Loss and Validation Loss')
            self.ax[0,0].set_xlabel('Step')
            self.ax[0,0].set_ylabel('Loss')
        
        if metrics['val_losses']:
            steps, losses = zip(*metrics['val_losses'])
            self.ax[0,0].plot(steps, losses, label = 'Validation Loss')
            self.ax[0,0].legend()
        
        # Training Learning Rate
        if metrics['train_lr']:       #.cpu().numpy()
            steps, accs = zip(*[(step, acc) for step, acc in metrics['train_lr']])
            self.ax[0,1].plot(steps, accs, label='Training Learning Rate')
            self.ax[0,1].set_title('Training Learning Rate')
            self.ax[0,1].set_xlabel('Step')
            self.ax[0,1].set_ylabel('Learning Rate')
            self.ax[0,1].legend()
        
        # Prediction Confidences
        if metrics['pred_confidences']:
            steps, confidences = zip(*metrics['pred_confidences'])
            self.ax[1,0].plot(steps, confidences, label='Prediction Confidences')
            self.ax[1,0].set_title('Prediciton Confidences')
            self.ax[1,0].set_xlabel('Step')
            self.ax[1,0].set_ylabel('Confidence')
            self.ax[1,0].legend()
        
        # Validation Accuracy
        if metrics['val_accs']:
            steps, accs = zip(*[(step, acc.cpu().numpy()) for step, acc in metrics['val_accs']])
            self.ax[1,1].plot(steps, accs, label='Validation Accuracy')
            self.ax[1,1].set_title('Validation Accuracy')
            self.ax[1,1].set_xlabel('Step')
            self.ax[1,1].set_ylabel('Accuracy')
            self.ax[1,1].legend()

        # Save the figure to a file
        plt.draw()
        plt.savefig(path_name)
        
    def has_zero_grad(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None and param.grad.abs().sum().item() != 0:
                    return False
        return True
    
    def check_initialization(self, model):
        print("Weight and Gradient Statistics:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"\n{name}:")
                # Weight stats
                print(f"Weight - Mean: {param.data.mean():.6f}")
                print(f"Weight - Std: {param.data.std():.6f}")
                print(f"Weight - Max: {param.data.max():.6f}")
                print(f"Weight - Min: {param.data.min():.6f}")
                
                # Check if weights follow approximately normal distribution
                weight_abs_mean = torch.abs(param.data.mean())
                weight_std = param.data.std()
                if weight_abs_mean > 0.1:
                    print(f"WARNING: Weights mean far from 0 ({weight_abs_mean:.6f})")
                if weight_std > 1.0:
                    print(f"WARNING: Weights std too large ({weight_std:.6f})")
                if weight_std < 0.01:
                    print(f"WARNING: Weights std too small ({weight_std:.6f})")

                # For attention layers, check scale
                if 'attn' in name and 'weight' in name:
                    expected_std = 1/math.sqrt(param.size(1))
                    if abs(weight_std - expected_std) > expected_std * 0.5:
                        print(f"WARNING: Attention weight std ({weight_std:.6f}) far from expected ({expected_std:.6f})")

    def train(self, epochs, batch_size, val_steps):
        print(f'Starting Training on Device: {self.device}...')
        self.model.train().to(self.device)          # Puts model on device and in train mode
        #self.check_initialization(self.model)

        self.optimizer.zero_grad()                  # Ensures optimizer starts with no gradients
        
        # Iterates through dataloader epochs number of times
        for epoch in tqdm(range(self.current_epoch, epochs), total=epochs):
            self.current_epoch = epoch      # Updates current epoch for checkpointing
            
            # Initializes persistent metric tracking across batches for effective batch metric
            running_loss = 0.0
            running_corrects = 0
            last_val_step = 0
            # Iterate through each batch in the dataloader
            for batch_idx, batch in enumerate(tqdm(self.dataloaders['train'], desc=f'Training Loop {epoch}')):
                # If initialized from checkpoint skip batches until at right batch_idx
                if epoch == self.current_epoch and batch_idx < self.current_batch_idx:
                    continue  # Skip batches processed in the previous run
                
                self.current_batch_idx = batch_idx  # Updates current batch_idx for checkpointing
                
                # Does a training step for current batch. 
                # Only backpropagates if reached accumulated steps or last effective batch otherwise only accumulates gradients
                running_loss, running_corrects, global_step, eff_step = self.train_step(batch, batch_idx, batch_size, epoch, running_loss, running_corrects)
                
                # # Makes sure model doesn't validate while first effective batch is still being completed
                not_last_val_step = last_val_step != eff_step
                
                # # Validates every val_steps effective batches. This checks if it has been val_steps.
                is_ready_to_validate = eff_step % val_steps == 0
                if (is_ready_to_validate and not_last_val_step) or batch_idx == len(self.dataloaders['train'])-1:
                    last_val_step = eff_step
                    self.val_step(batch_idx, global_step, eff_step, epoch)  # Validates model
                    self.model.train() # Sets model back to train mode (model put into evalution mode in val_step function)

                # Update plots after each training loss step (short and worth doing for easy live tracking)
                self.update_plots(self.metrics)

            self.current_batch_idx = 0  # Resets current batch after epoch is done for checkpointing
            if epoch == 1:
                checkpoint = {
                                'epoch': self.current_epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'batch_idx': self.current_batch_idx,
                                'metrics': self.metrics
                            }
                save_checkpoint(checkpoint, self.epoch_checkpoint_path)
            

        return self.metrics
    
    def train_step(self, batch, batch_idx, batch_size, epoch, running_loss, running_corrects):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)   # Put inputs and targets on device
        
        def check_device_consistency(*tensors, expected_device=self.device):
            for i, t in enumerate(tensors):
                if hasattr(t, 'device') and t.device != expected_device:
                    raise RuntimeError(f"Tensor {i} on wrong device. Expected {expected_device}, got {t.device}")
        
        # Check model device
        model_devices = {p.device for p in self.model.parameters()}
        if len(model_devices) > 1:
            raise RuntimeError(f"Model parameters spread across devices: {model_devices}")
        if next(self.model.parameters()).device != torch.device(self.device):
            raise RuntimeError(f"Model on {next(self.model.parameters()).device}, expected {self.device}")
        
        # Check all tensors
        check_device_consistency(
            inputs, 
            targets,
            next(self.model.parameters()),
            expected_device=torch.device(self.device)
        )
        
        if torch.cuda.is_available():
            if torch.cuda.memory_allocated() > 0.95 * torch.cuda.get_device_properties(0).total_memory:
                raise RuntimeError("GPU memory usage too high")
        
        if not torch.is_tensor(inputs) or not torch.is_tensor(targets):
            raise ValueError(f"Inputs/targets must be tensors. Got {type(inputs)}/{type(targets)}")
        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            raise ValueError("NaN values detected in inputs/targets")
        
        # Checks to see if needs to handle shorter last accumulated batch
        is_divisible = len(self.dataloaders['train']) % self.accumulated_steps == 0
        
        # checks if batch_idx is part of last effective batch i.e. len(dataloaders) = 32 accum_steps = 5, then would be checking if batch_idx + 1 > 32-(32%5) = 30
        is_last_batch = batch_idx + 1 > (len(self.dataloaders['train']) - (self.accumulated_steps if is_divisible else len(self.dataloaders['train']) % self.accumulated_steps))
        
        # Gets remaining number of steps that aren't a perfect number of self.accumulated_steps if last batch of not divisible data loader
        actual_accumulation = len(self.dataloaders['train']) % self.accumulated_steps if is_last_batch and not is_divisible else self.accumulated_steps
        
        # Calculates Effective Step for current batch for scheduler step 
        eff_step = (batch_idx+1)//self.accumulated_steps + (1 if is_last_batch and not is_divisible else 0)
        
        # Forward pass
        with autocast(dtype=torch.bfloat16):
            outputs = self.model(inputs)
            # Each loss is scaled by number of accumulated steps in effective batch
            loss = self.criterion(outputs, targets) / actual_accumulation
        
        if torch.isnan(outputs).any():
            raise ValueError("NaN values in model outputs")
        if not outputs.size(1) == self.model.num_class:  # Add num_classes as class attribute
            raise ValueError(f"Unexpected output shape: {outputs.shape}")

        # calculate prediction based on highest probability
        _, preds = torch.max(outputs, 1)
        
        # Adds loss to existing running loss. Descales loss item based on number of steps in effective batch
        new_running_loss = running_loss + (loss.item() * actual_accumulation)
        
        # Adds number of correct predicitons in batch to running corrects
        new_running_corrects = running_corrects + torch.sum(preds == targets.data)
        
        # Accumulates gradients
        loss.backward()
        
        # Checks for NaNs in gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm):
                    print(f"NaN gradient detected in {name}")
                    raise ValueError("NaN gradient detected")
        
        # Checks to see if this is the last batch in the dataloader
        is_true_last_batch = batch_idx == len(self.dataloaders['train'])-1
        
        # Checks to see if it has accumulated desired number of steps or is the last effective batch
        if (batch_idx + 1) % self.accumulated_steps == 0 or is_true_last_batch:
            # Gradient clipping
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            
            # # Raises an error so I can fix if gradients are not clipped properly
            # if not torch.isfinite(grad_norm):
            #     self.optimizer.zero_grad()
            #     raise ValueError(f"Gradients are not finite. Norm: {grad_norm}")
            
            grad_norm = torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])
        

            print(f"\nMax gradient norm: {grad_norm.max()}")
            print(f"Min gradient norm: {grad_norm.min()}")
            print(f"Mean gradient norm: {grad_norm.mean()}")
            print(f"Std gradient norm: {grad_norm.std()}")
            
            if grad_norm.max() > 10.0:  # Adjust threshold based on monitoring
                print("Warning: Large gradients detected")
            # Changes weights
            self.optimizer.step()
            # Resets gradients for next accumulation step
            self.optimizer.zero_grad()
            
            # iters is set so it can handle a dataloader that is not perfectly divisible by accumulated steps so need to handle fraction properly
            # only modifies if len of dataloaders in not perfectly divisble by accumulated steps and is the last batch in dataloader
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-8:  # Adjust threshold as needed
                print(f"Warning: Learning rate very low: {current_lr}")
            
            # Calculates effective global step for plotting purposes
            global_step = self.calculate_step(batch_idx, epoch)
            
            # Calculates effective batch metrics for plotting
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0].mean().item()
            acc = running_corrects.double() / (actual_accumulation * batch_size)
            loss_avg = running_loss / actual_accumulation
            
            # Prints metrics for monitoring
            print_metrics(self.iters,
                          epoch, 
                          eff_step,
                          loss_avg, 
                          acc, 
                          None, 
                          None
                          )
            
            # Logs Metrics for Tracking
            self.metrics['train_losses'].append((global_step, loss_avg))
            self.metrics['train_lr'].append((global_step, self.optimizer.param_groups[0]['lr']))
            self.metrics['pred_confidences'].append((global_step, confidences))

            # resets running_loss and running_corrects for next effective batch. 
            new_running_loss = 0
            new_running_corrects = 0
        else:
            global_step = None
            
        return new_running_loss, new_running_corrects, global_step, eff_step
            
             
    def val_step(self, batch_idx, global_step, eff_step, epoch):
        print('Starting Validation...')
        # Performs evaluation. Model put into evaluate mode in evaluate_model function
        val_loss, val_acc, val_precision, val_recall = evaluate_model(self.model, self.dataloaders['val'], self.criterion, self.device)

        # Logs Metrics for Tracking
        self.metrics['val_losses'].append((global_step, val_loss))
        self.metrics['val_accs'].append((global_step, val_acc))
        self.metrics['val_precisions'].append((global_step, val_precision))
        self.metrics['val_recalls'].append((global_step, val_recall))
        
        # prints metrics for monitoring
        print_metrics(self.iters,
                      epoch,
                      eff_step,
                      val_loss,
                      val_acc,
                      val_precision,
                      val_recall,
                      phase='Validation')
        
        # Save the best checkpoint based on validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                            'epoch': self.current_epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'batch_idx': self.current_batch_idx,
                            'metrics': self.metrics
                        }
            save_checkpoint(checkpoint, self.best_checkpoint_path)
    
    def calculate_step(self, batch_idx, epoch):
        # Helper function to calculate effective global step based on batch index and epoch
        
        # Calculates number of steps per epoch, if not perfectly divisible by accumulated steps, adds one for remaining effective batch
        step_factor = (len(self.dataloaders['train']) // self.accumulated_steps) + (1 if len(self.dataloaders['train']) % self.accumulated_steps != 0 else 0)
        # adds one if this is the last batch in the dataloader
        offset = 1 if (batch_idx == len(self.dataloaders['train'])-1) and (len(self.dataloaders['train']) % self.accumulated_steps != 0) else 0
        # Calculates effective step
        return step_factor * epoch + (batch_idx + 1) // self.accumulated_steps + offset