import torch 
import torch.nn as nn 

class ConditionalTemporalGenerator(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

        # Adjust the first layer to account for the additional input size from the label vector
        self.model = nn.Sequential(
            nn.ConvTranspose1d(100 + embed_size, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 100, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Initialize weights
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, z, l):
        # Concatenate the noise vector z and the one-hot encoded label vector l
        l = l.view(l.size(0), -1, 1)  # Ensure l has shape (batch_size, num_labels, 1)
        z = z.view(z.size(0), -1, 1)  # Ensure z has shape (batch_size, 100, 1)
        x = torch.cat((z, l), dim=1)  # Concatenate along the channel dimension
        
        # Apply the model and reshape the output
        x = self.model(x).transpose(1, 2) 
        return x 
    
class ConditionalVideoGenerator(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        
        # Instantiate the conditional temporal generator
        self.temp = ConditionalTemporalGenerator(embed_size)

        # Create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # Create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        ) 

        # Define the image generator 
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512+embed_size, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Initialize weights
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, z, l): 
        batch_size = l.size(0)
        embed_size = l 
        embed_size = embed_size.unsqueeze(2).unsqueeze(3).repeat(1,1,4,4) 
        
        # Pass our latent vector and label through the temporal generator and reshape
        z_fast = self.temp(z, l).contiguous()
        z_fast = z_fast.view(-1, 100)

        # transform the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        z_slow = self.slow(z).view(-1, 256, 4, 4)
        # Concatenate z_slow with embed dimension
        z_slow = torch.cat([z_slow,embed_size],dim=1).unsqueeze(1) 
        
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*16, dim=1).view(-1, 256+768, 4, 4)
        
        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1) 
  
        # Transform into image frames
        out = self.model(z)

        return out.view(-1, 16, 3, 64, 64).transpose(1, 2)

class ConditionalCritic(nn.Module):
    def __init__(self, nc, ngf, embed_size, image_size):
        super(ConditionalCritic, self).__init__()
        self.image_size = image_size

        # Linear layer to transform text embedding
        self.embed = nn.Linear(embed_size, image_size * image_size * 16)

        # Define the 3D convolutional layers
        self.main3d = nn.Sequential(
            nn.Conv3d(3 + 1, ngf, kernel_size=4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf, ngf * 2, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 2, ngf * 4, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 4, ngf * 8, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Define the final 2D convolutional layer
        self.conv2d = nn.Conv2d(ngf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

        # Initialize weights
        self.main3d.apply(self.init_weights)
        self.init_weights(self.conv2d)
        self.init_weights(self.embed)

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=2**0.5)

    def forward(self, input, text_embedding):
        # Transform and reshape text embedding
        # Check the shape after initial transformation
        embedding = self.embed(text_embedding)
        
        # Reshape to match video tensor dimensions
        embedding = embedding.view(text_embedding.size(0), 1, 16, self.image_size, self.image_size)

        # Expand to match input dimensions
        embedding = embedding.expand(-1, -1, input.size(2), -1, -1)
        
        # Concatenate along the channel dimension
        input = torch.cat([input, embedding], dim=1)

        # Pass the concatenated tensor through the 3D convolutional layers
        h = self.main3d(input)

        # Reshape the tensor for the 2D convolutional layer
        h = h.view(h.size(0), -1, h.size(3), h.size(4))

        # Pass the tensor through the final 2D convolutional layer
        h = self.conv2d(h)

        return h 

def singular_value_clip(w):
    dim = w.shape
    # reshape into matrix if not already MxN
    if len(dim) > 2:
        w = w.reshape(dim[0], -1)
    u, s, v = torch.svd(w, some=True)
    s[s > 1] = 1
    return (u @ torch.diag(s) @ v.t()).view(dim) 