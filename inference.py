import numpy as np
import os
import re
import torch
from transformers import BertTokenizer, BertModel 
from moviepy.editor import ImageSequenceClip 
from main import save_dir, ConditionalVideoGenerator,embed_size,device 

netG_inf = ConditionalVideoGenerator(embed_size) 
labels = ["blue_fireball2"]

def get_latest_checkpoint(save_dir):
    # Get all files in the save directory
    files = os.listdir(save_dir)
    
    # Regular expression to match the checkpoint files with the pattern "{epoch}.pth"
    checkpoint_pattern = re.compile(r"(\d+)\.pth")
    
    # Extract epoch numbers from filenames
    epochs = []
    for file in files:
        match = checkpoint_pattern.match(file)
        if match:
            epochs.append(int(match.group(1)))

    # If no checkpoint files found, return None
    if not epochs:
        print("No checkpoint files found.")
        return None
    
    # Get the largest epoch number
    max_epoch = max(epochs)
    
    # Construct the file path for the latest checkpoint
    checkpoint_path = os.path.join(save_dir, f"{max_epoch}.pth")
    
    return checkpoint_path, max_epoch

def load_netG_from_checkpoint(checkpoint_path, netG):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load only the netG state dict
    netG.load_state_dict(checkpoint['netG_state_dict'])
    print(f"Loaded netG from checkpoint at epoch {checkpoint['epoch']}")

checkpoint_path, max_epoch = get_latest_checkpoint(save_dir)

if checkpoint_path:
    print(f"Loading netG from checkpoint at epoch {max_epoch} located at {checkpoint_path}")
    load_netG_from_checkpoint(checkpoint_path, netG_inf)
else:
    print("No checkpoint found, starting from scratch.")

def genSingleSample(g, embedding, device='cuda'):
    '''
    Generate a single video for a given embedding using generator g
    ''' 
    g.eval()
    b_size = embedding.size(0)
    with torch.no_grad(): 
        z = torch.rand((b_size, 100), device=device) * 2 - 1
        s = g(z, embedding).cpu().detach().numpy()

    # The generated video has a shape (b_size, channels, frames, height, width) 
    out = s[0]  # Get the first generated video
    out = out.transpose((1, 2, 3, 0))  # Transpose to (frames, height, width, channels)
    out = (out + 1) / 2 * 255  # Normalize to [0, 255]
    out = out.astype(np.uint8)  # Convert to uint8
    clip = ImageSequenceClip(list(out), fps=20)
    clip.write_gif('single_sample_rgb.gif', fps=20)

def get_text_embedding(label, tokenizer, model):
    inputs = tokenizer(label, return_tensors='pt')
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze(0) 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

label_embeddings = torch.stack([get_text_embedding(label, tokenizer, bert_model) for label in labels]).to(device)
single_label_embedding = label_embeddings[0].unsqueeze(0)  # Extract the first embedding and add batch dimension
genSingleSample(netG_inf, single_label_embedding)