import torch 
from transformers import BertTokenizer, BertModel 
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.videos = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Initialize BERT tokenizer and model (not needed if we only want the raw labels)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.videos)

    def get_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():  # Ensure no gradients are calculated for embeddings
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Get the average embedding of the last hidden state
        return embedding.squeeze(0)  # Remove the batch dimension

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.videos[idx])
        frames = sorted(os.listdir(video_path))  # Ensure frames are in the correct order
        sequence = []

        for frame in frames:
            frame_path = os.path.join(video_path, frame)
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence.append(image)

        sequence = torch.stack(sequence, dim=0)  # Shape: (num_frames, channels, height, width)
        sequence = sequence.permute(1, 0, 2, 3)  # Change to (channels, num_frames, height, width)

        # Get the class label (folder name)
        class_label = self.videos[idx]

        return sequence, class_label