import os 
import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel 
from PIL import Image
import imageio
import ipywidgets as widgets
from IPython.display import display
import base64
from frame_dataset import FrameDataset
from dataset import resize_animation,rearrange_frames_in_directory
from model import ConditionalTemporalGenerator,ConditionalCritic,ConditionalVideoGenerator,singular_value_clip

# Directory containing all animations
animations_directory = 'Fireball_animations'

# Directory containing all resized animations
resized_animations_directory = 'Fireball_Resized' 

#Checkpoint path 
checkpoint_filename = "model_checkpoint.pth"

# Get the current directory path
current_directory = os.getcwd()

# Combine the current directory path with the checkpoint filename
checkpoint_path = os.path.join(current_directory, checkpoint_filename)

nz = 100 #random noise vector
nc = 3 #channel dimension
ngf = 64 #image dimension
image_size = 64 
ncf = 64 
lr = 0.00005
num_epochs = 10000
embed_size = 768 #Fixed value incase of using the BERT Model for embeddings 
lambda_gp = 10 
device = torch.device('cuda')
critic_iter = 5
batch_size = 4 

# Process each animation directory
for animation_folder in os.listdir(animations_directory):
    animation_folder_path = os.path.join(animations_directory, animation_folder)
    if os.path.isdir(animation_folder_path):
        resize_animation(animation_folder_path)

# Process each resized animation directory
for animation_folder in os.listdir(resized_animations_directory):
    animation_folder_path = os.path.join(resized_animations_directory, animation_folder)
    if os.path.isdir(animation_folder_path):
        rearrange_frames_in_directory(animation_folder_path)

print("Frames rearranged in a uniform manner.")


root_dir = resized_animations_directory
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FrameDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Check the first batch to diagnose potential issues
for data, class_labels in dataloader:
    print(f'Data shape: {data.shape}')  # Expected: (batch_size, channels, num_frames, height, width)
    print(f'Class labels: {class_labels}')  # Expected: list of class labels
    break 


netC = ConditionalCritic(nc,ngf,embed_size,image_size).to(device)
netG = ConditionalVideoGenerator(embed_size).to(device)


optimizerC = optim.RMSprop(netC.parameters(),lr = lr)
optimizerG = optim.RMSprop(netG.parameters(),lr = lr)


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get text embeddings from labels
def get_text_embedding(label, tokenizer, model):
    inputs = tokenizer(label, return_tensors='pt')
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze(0)

# Function to generate and display samples
def genSamples(g, embeddings, device='cuda'):
    '''
    Generate videos, given a generator g
    '''
    b_size = embeddings.size(0)
    with torch.no_grad(): 
        z = torch.rand((b_size, 100), device=device) * 2 - 1
        s = g(z, embeddings).cpu().detach().numpy()

    grid_size = int(np.ceil(np.sqrt(b_size)))
    out = np.zeros((3, 16, 64*grid_size, 64*grid_size))

    for j in range(grid_size):
        for k in range(grid_size):
            idx = j * grid_size + k
            if idx < b_size:
                out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[idx, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    out = out.astype(np.uint8)

    frames = [Image.fromarray(frame) for frame in out]
    gif_path = "/tmp/sample.gif"
    imageio.mimsave(gif_path, frames, fps=20)
    return gif_path

# Function to display the GIF with looping
def display_gif_with_label(gif_path, label):
    label_widget = widgets.Label(value=f"Label: {label}")
    with open(gif_path, "rb") as f:
        gif_data = f.read()
    gif_base64 = base64.b64encode(gif_data).decode('utf-8')
    gif_html = f'<img src="data:image/gif;base64,{gif_base64}" style="max-width:100%; width:auto; height:auto;" loop="true" />'
    gif_widget = widgets.HTML(value=gif_html)
    display(widgets.VBox([label_widget, gif_widget]))

print("Starting the loop......")
# Assuming netG, netC, optimizerG, optimizerC, dataloader, num_epochs are defined elsewhere
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader, 0):

        # Get text embeddings for labels
        label_embeddings = torch.stack([get_text_embedding(label, tokenizer, bert_model) for label in labels]).to(device)

        # Train Discriminator
        optimizerC.zero_grad()
        real = data.to(device)
        b_size = real.size(0)

        # Update discriminator 
        for _ in range(1):
            
            pr = netC(real, label_embeddings)
            z = torch.rand((b_size, 100), device=device) * 2 - 1
            fake = netG(z, label_embeddings)
            pf = netC(fake.detach(), label_embeddings)
            dis_loss = torch.mean(-pr) + torch.mean(pf)
            dis_loss.backward(retain_graph=True)
            optimizerC.step()

        # Train Generator
        optimizerG.zero_grad()
        z = torch.rand((b_size, 100), device=device) * 2 - 1
        fake = netG(z, label_embeddings) 
        pf = netC(fake, label_embeddings)
        gen_loss = torch.mean(-pf)
        gen_loss.backward()
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     dis_loss.item(), gen_loss.item()))

    # Display generator output every 50 epochs
    if epoch % 5 == 0:
        gif_path = genSamples(netG, label_embeddings)
        label = labels[0]  # Assuming you want to display the first label
        display_gif_with_label(gif_path, label)

    # Enforce 1-Lipschitz constraint
    if epoch % 5 == 0:
        for module in list(netC.main3d.children()) + [netC.conv2d]:
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                module.weight.data = singular_value_clip(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                gamma = module.weight.data
                std = torch.sqrt(module.running_var)
                gamma[gamma > std] = std[gamma > std]
                gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                module.weight.data = gamma

# Function to save the training state
def save_checkpoint(epoch, netG, netC, optimizerG, optimizerC, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netC_state_dict': netC.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerC_state_dict': optimizerC.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

save_checkpoint(epoch, netG, netC, optimizerG, optimizerC,checkpoint_path)

# Function to load the training state
def load_checkpoint(checkpoint_path, netG, netC, optimizerG, optimizerC):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netC.load_state_dict(checkpoint['netC_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerC.load_state_dict(checkpoint['optimizerC_state_dict'])
    epoch = checkpoint['epoch']
    lossG = checkpoint['lossG']
    lossC = checkpoint['lossC']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, lossG, lossC