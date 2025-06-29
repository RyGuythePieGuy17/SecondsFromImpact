# Made by Ryan Geisen
import torch, os, math
from tqdm import tqdm

#Eval function for readability 
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_corrects = 0
    TP=TN=FP=FN=0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluation Loop'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == targets.data)
            # Calculate TP, TN, FP, FN
            TP += ((preds == 1) & (targets == 1)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = running_corrects.double() / len(dataloader.dataset)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return avg_loss, accuracy, precision, recall


#Print Metrics function for readability
def print_metrics(tot_steps, epoch, batch_idx, loss_avg, acc, precision, recall, phase='Training'):
    if phase == 'Validation':
        print(f'{phase} - Epoch: {epoch + 1} Batch: {batch_idx}/{tot_steps} Loss: {loss_avg:.4f} Accuracy: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f}')
    else:
        print(f'{phase} - Epoch: {epoch + 1} Batch: {batch_idx}/{tot_steps} Loss: {loss_avg:.4f} Accuracy: {acc:.4f}')


def save_checkpoint(state, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    torch.save(state, filename)
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)  # Move optimizer state to the device
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx'] if 'batch_idx' in checkpoint else 0  # Load the last batch index if available
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch: {checkpoint['epoch']}, batch: {start_batch_idx})")
        return start_epoch, checkpoint['metrics'], start_batch_idx, model
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, None

def load_vit_weights(to_model_dict, from_model_dict, dim, SEED):
    # Load the weights from the ViT model
    generator = torch.Generator()
    generator.manual_seed(SEED)
    key_mapping = {
        'attn.in_proj_weight' : None,
        'attn.in_proj_bias':None,
        'out_proj.weight': 'attention.output.dense.weight',
        'out_proj.bias':'attention.output.dense.bias',
        'norm1.weight': 'layernorm_before.weight',
        'norm1.bias': 'layernorm_before.bias',
        'norm2.weight': 'layernorm_after.weight',
        'norm2.bias': 'layernorm_after.bias',
        '0.weight': 'intermediate.dense.weight',
        '0.bias': 'intermediate.dense.bias',
        '3.weight': 'output.dense.weight',
        '3.bias': 'output.dense.bias'
    }
        
    for key in to_model_dict.keys():
        if 'spatial_encStack' not in key:
            # Initialize new weights based on their type
            if 'temporal_embedding.temporal_pos_embedding' in key:
                to_model_dict[key] = torch.randn_like(
                    to_model_dict[key],
                    generator=generator
                    ) * 0.02
            
            elif 'temporal_encStack' in key:
                if 'attn' in key and 'weight' in key:
                    gain = 1/math.sqrt(2)
                    to_model_dict[key] = torch.nn.init.xavier_uniform_(
                        torch.empty_like(to_model_dict[key]),
                        gain=gain,
                        generator=generator)
                elif 'attn' in key and 'bias' in key:
                    to_model_dict[key] = torch.zeros_like(to_model_dict[key])
                elif 'ffn' in key and 'weight' in key:
                    to_model_dict[key] = torch.nn.init.xavier_uniform_(
                        torch.empty_like(to_model_dict[key]),
                        generator=generator)
                elif 'ffn' in key and 'bias' in key:
                    to_model_dict[key] = torch.zeros_like(to_model_dict[key])
            
            elif 'MLP_head' in key:
                if '2.weight' in key:  # First linear layer
                    gain = 1/math.sqrt(2)
                    to_model_dict[key] = torch.nn.init.xavier_uniform_(
                        torch.empty_like(to_model_dict[key]), 
                        gain=gain,
                        generator=generator)
                elif '2.bias' in key:
                    to_model_dict[key] = torch.zeros_like(to_model_dict[key])
                elif '6.weight' in key:  # Final layer
                    to_model_dict[key] = torch.nn.init.xavier_uniform_(
                        torch.empty_like(to_model_dict[key]),
                        generator=generator)
                elif '6.bias' in key:
                    to_model_dict[key] = torch.zeros_like(to_model_dict[key])
        else:
            suffix = '.'.join(key.split('.')[-2:])
            mapped_suffix = key_mapping[suffix]
            layer_number = key.split('.')[1]
   
            if mapped_suffix == None:
                if suffix.endswith('bias'):
                    q_bias = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.query.bias'][:dim]
                    k_bias = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.key.bias'][:dim]
                    v_bias = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.value.bias'][:dim]
                    
                    weights = torch.cat([q_bias, k_bias, v_bias], dim = 0) 
                else:
                    q_weight = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.query.weight'][:dim, :dim]
                    k_weight = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.key.weight'][:dim, :dim]
                    v_weight = from_model_dict[f'encoder.layer.{layer_number}.attention.attention.value.weight'][:dim, :dim]
                    
                    weights = torch.cat([q_weight, k_weight, v_weight], dim = 0)
            else:
                weights = from_model_dict[f'encoder.layer.{layer_number}.{mapped_suffix}']
            
            to_model_dict[key] = weights
    
    return to_model_dict
    
            
def move_to_cpu(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.cpu()
        elif isinstance(value, dict):
            move_to_cpu(value)
    return state_dict