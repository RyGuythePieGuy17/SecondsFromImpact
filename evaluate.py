# Standard Imports
import torch, math
import numpy as np
import time
import os
from PIL import Image

from torch import nn
from transformers import ViTModel
from torch.utils.data import random_split, Subset

# Custom Imports
from train_class import Trainer
from vivit import ViVit
#from factenc_vivit import ViVit, SemiCon_ViVit
from train_utils import load_vit_weights
from dataset_class import Custom_Traffic_Dataset
from tqdm import tqdm

def save_frame_sequence(inputs, pred, target, counter, model_num):
    """
    Save a sequence of frames as a single image with appropriate FP/FN label
    
    Args:
        inputs: torch tensor of shape (batch_size, channels, frames, height, width) or (batch_size, frames, channels, height, width)
        pred: predicted class (0 or 1)
        target: true class (0 or 1) 
        counter: unique identifier for filename
    """
    # Remove batch dimension since batch_size=1
    frames = inputs.squeeze(0)  # Now shape is (channels, frames, H, W) or (frames, channels, H, W)
    frames = frames.unsqueeze(1)

    # Handle different input formats
    if frames.dim() == 4:
        if frames.shape[0] == 3 or frames.shape[0] == 1:  # (C, T, H, W)
            frames = frames.permute(1, 0, 2, 3)  # Convert to (T, C, H, W)
        # else assume it's already (T, C, H, W)
    
    # Convert to numpy and handle normalization
    frames_np = frames.cpu().numpy()
    frames_np = frames_np - frames_np.min()
    frames_np = ((frames_np / frames_np.max()) * 255).astype(np.uint8)
    
    # Create a grid of frames
    num_frames = frames_np.shape[0]
    
    # Calculate grid dimensions (try to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(num_frames)))
    grid_rows = int(np.ceil(num_frames / grid_cols))
    
    # Get frame dimensions
    if frames_np.shape[1] == 1:  # Grayscale
        frame_h, frame_w = frames_np.shape[2], frames_np.shape[3]
        is_grayscale = True
    else:  # RGB
        frame_h, frame_w = frames_np.shape[2], frames_np.shape[3]
        is_grayscale = False
    
    # Create the combined image
    if is_grayscale:
        combined_img = np.zeros((grid_rows * frame_h, grid_cols * frame_w), dtype=np.uint8)
    else:
        combined_img = np.zeros((grid_rows * frame_h, grid_cols * frame_w, 3), dtype=np.uint8)
    
    # Place each frame in the grid
    for i, frame in enumerate(frames_np):
        row = i // grid_cols
        col = i % grid_cols
        
        start_row = row * frame_h
        end_row = start_row + frame_h
        start_col = col * frame_w
        end_col = start_col + frame_w
        
        if is_grayscale:
            combined_img[start_row:end_row, start_col:end_col] = frame[0]
        else:
            combined_img[start_row:end_row, start_col:end_col] = frame.transpose(1, 2, 0)
    
    # Convert to PIL Image
    if is_grayscale:
        pil_img = Image.fromarray(combined_img, mode='L')
    else:
        pil_img = Image.fromarray(combined_img, mode='RGB')
    
    # Determine classification type
    if pred == 1 and target == 0:
        classification_type = "FP"
    elif pred == 0 and target == 1:
        classification_type = "FN"
    else:
        return  # Not a misclassification, don't save
    
    # Create filename
    filename = f"./results/models/model{model_num}/misclassifications/{classification_type}_{counter:04d}_pred{pred}_true{target}.png"
    
    # Save the image
    pil_img.save(filename)
    print(f"Saved misclassification: {filename}")

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
    batch_size = 1        # batch size
    
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
    accumulated_steps = 40 # number of forward passes before updating weights (Effective batch size = batch_size * accumulated_steps)
    #TODO: Reapply regularization
    weight_decay = 0 # 1e-4        # Weight Decay
    dropout = 0.2           # Dropout
    val_steps = 1000

    #File Management Parameters
    model_num = 2
    load_from_checkpoint = False
    load_checkpoint_path = f'./model28/checkpoint.pth'
    out_checkpoint_path = f'./results/models/model{model_num}/checkpoint.pth'
    best_checkpoint_path = f'./results/models/model{model_num}/best_checkpoint.pth'
    epoch_checkpoint_path = f'./results/models/model{model_num}/first_epoch_checkpoint.pth'
    
    # Create directory for saving misclassifications
    os.makedirs(f"./results/models/model{model_num}/misclassifications", exist_ok=True)

    # Counter for unique filenames
    misclassification_counter = 0
    
    ###Testing Params
    interval = 16
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
    
    # Load ViT weights into spatial encoder
    model_dict = torch.load(f'./results/models/model{model_num}/best_checkpoint.pth')
    model_dict = model_dict['model_state_dict']
    model.load_state_dict(model_dict, strict=True)
    del model_dict
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
    
    print("Evaluating...")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        max_time = avg_time = 0
        min_time = 99999
        running_corrects = 0
        TP = TN = FP = FN = 0
        pbar = tqdm(dataloaders['test'])
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.time()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            end_time = time.time() - start_time
            running_corrects += torch.sum(preds == targets.data)
            if end_time > max_time:
                max_time = end_time
            if end_time < min_time:
                min_time = end_time
            avg_time += end_time
            # Calculate TP, TN, FP, FN
            TP += ((preds == 1) & (targets == 1)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()
            
            # # Save misclassifications
            # pred_val = preds.item()  # Since batch_size=1
            # target_val = targets.item()
            
            # if pred_val != target_val:  # Misclassification detected
            #     save_frame_sequence(inputs, pred_val, target_val, misclassification_counter, model_num)
            #     misclassification_counter += 1
            
             # Update the progress bar description with current accuracy
            cur_accuracy = running_corrects / ((pbar.n + 1) * batch_size)
            pbar.set_description(f"Accuracy: {cur_accuracy:.4f}")
    
    
    accuracy = running_corrects.double() / len(dataloaders['test'].dataset)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    avg_pred_time = avg_time / len(dataloaders['test'].dataset)
    
    print('\n')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Avg Time: {avg_pred_time}')
    print(f'Max Time: {max_time}')
    print(f'Min Time: {min_time}')

if __name__ == '__main__':
    main()