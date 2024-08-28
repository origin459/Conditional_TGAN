import torch

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