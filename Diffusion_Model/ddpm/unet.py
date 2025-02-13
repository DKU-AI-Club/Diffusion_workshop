import torch
from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        """Basic Block module in UNet, including time embedding and up/downsampling functionality"""
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First convolution
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Inject time information into feature maps
        h = h + time_emb[..., None, None]
        # Second convolution
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Upsampling or downsampling
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    """Embed time steps using sinusoidal positional encoding, inspired by the positional encoding in Transformers.
    Uses sine and cosine functions to map time steps into a high-dimensional space."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        # Split the dimension into two halves for sin and cos
        half_dim = self.dim // 2
        # Compute exponential decay for different frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        # Generate the frequency sequence
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Multiply the time step with the frequency sequence
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sin and cos to form the final embedding vector
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """A simple UNet model for noise prediction in diffusion models."""
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding layer
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input layer, downsampling layers, upsampling layers, and output layer
        self.input = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x, time_step):
        # Time step embedding
        t = self.time_embed(time_step)
        # Initial convolution
        x = self.input(x)
        # UNet forward pass: first downsample to collect features, then upsample to restore resolution
        residual_stack = []
        for down in self.downs:
            x = down(x, t)
            residual_stack.append(x)
        for up in self.ups:
            residual_x = residual_stack.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

def print_shapes(model, x, time_step):
    print("Input shape:", x.shape)
    
    # Time step embedding
    t = model.time_embed(time_step)
    print("Time embedding shape:", t.shape)
    
    # Initial convolution
    x = model.input(x)
    print("After input conv shape:", x.shape)
    
    # Downsampling process
    residual_stack = []
    print("\nDownsampling process:")
    for i, down in enumerate(model.downs):
        x = down(x, t)
        residual_stack.append(x)
        print(f"Down block {i+1} output shape:", x.shape)
    
    # Upsampling process
    print("\nUpsampling process:")
    for i, up in enumerate(model.ups):
        residual_x = residual_stack.pop()
        x = torch.cat((x, residual_x), dim=1)
        print(f"Concatenated input shape before up block {i+1}:", x.shape)
        x = up(x, t)
        print(f"Up block {i+1} output shape:", x.shape)
    
    # Final output
    output = model.output(x)
    print("\nFinal output shape:", output.shape)
    return output


if __name__ == "__main__":
    model = SimpleUnet()
    x = torch.randn(1, 3, 32, 32)
    time_step = torch.tensor([10])
    print_shapes(model, x, time_step)
