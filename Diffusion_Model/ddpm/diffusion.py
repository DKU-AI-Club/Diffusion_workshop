import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000):
        """Initialize the noise scheduler
        Args:
            beta_start: β1, initial noise level
            beta_end: βT, final noise level  
            num_steps: T, number of diffusion steps
            device: Running device
        """
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps

        # β_t: Linear noise schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_steps))
        # α_t = 1 - β_t 
        self.register_buffer('alphas', 1.0 - self.betas)
        # α_bar_t = ∏(1-β_i) from i=1 to t
        self.register_buffer('alpha_bar', torch.cumprod(self.alphas, dim=0))
        # α_bar_(t-1)
        self.register_buffer('alpha_bar_prev', torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0))
        # sqrt(α_bar_t)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
        # 1/sqrt(α_t)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))
        # sqrt(1-α_bar_t)
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - self.alpha_bar))

        # 1/sqrt(α_bar_t)
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1.0 / self.alpha_bar))
        # sqrt(1/α_bar_t - 1)
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1.0 / self.alpha_bar - 1))
        # Posterior variance σ_t^2
        self.register_buffer('posterior_var', self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        # Posterior mean coefficient 1: β_t * sqrt(α_bar_(t-1))/(1-α_bar_t)
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        # Posterior mean coefficient 2: (1-α_bar_(t-1)) * sqrt(α_t)/(1-α_bar_t)
        self.register_buffer('posterior_mean_coef2', (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar))
    
    def get(self, var, t, x_shape):
        """Retrieve the value of a variable at a specific timestep and adjust its shape
        Args:
            var: The variable to query
            t: Timestep
            x_shape: Target shape
        Returns:
            Adjusted variable value
        """
        # Retrieve the value at the specified timestep from the variable tensor
        out = var[t]
        # Reshape to [batch_size, 1, 1, 1] for broadcasting
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    def add_noise(self, x, t):
        """Add noise to the input
        Implements the formula: x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε, ε ~ N(0,I)
        Args:
            x: Input image x_0
            t: Timestep
        Returns:
            (noisy_x, noise): Noised image and the noise used
        """
        # Retrieve sqrt(α_bar_t) for timestep t
        sqrt_alpha_bar = self.get(self.sqrt_alpha_bar, t, x.shape)
        # Retrieve sqrt(1-α_bar_t) for timestep t
        sqrt_one_minus_alpha_bar = self.get(self.sqrt_one_minus_alpha_bar, t, x.shape)
        # Sample noise from standard normal distribution ε ~ N(0,I)
        noise = torch.randn_like(x)
        # Implement the forward diffusion process: x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


def plot_diffusion_steps(image, noise_scheduler, step_size=100):
    """Plot the process of gradually adding noise to an image
    Args:
        image: Original image
        noise_scheduler: Noise scheduler
        step_size: Interval of steps to plot
    Returns:
        fig: The plotted figure
    """
    num_images = noise_scheduler.num_steps // step_size
    fig = plt.figure(figsize=(15, 3))
    
    # Plot the original image
    plt.subplot(1, num_images + 1, 1)
    plt.imshow(show_tensor_image(image))
    plt.axis('off')
    plt.title('Original')
    
    # Plot noisy images at different timesteps
    for idx in range(num_images):
        t = torch.tensor([idx * step_size])
        noisy_image, _ = noise_scheduler.add_noise(image, t)
        plt.subplot(1, num_images + 1, idx + 2)
        plt.imshow(show_tensor_image(noisy_image))
        plt.axis('off')
        plt.title(f't={t.item()}')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader import load_transformed_dataset, show_tensor_image

    train_loader, test_loader = load_transformed_dataset()
    image, _ = next(iter(train_loader))
    noise_scheduler = NoiseScheduler()
    noisy_image, noise = noise_scheduler.add_noise(image, torch.randint(0, noise_scheduler.num_steps, (image.shape[0],)))
    plt.imshow(show_tensor_image(noisy_image))

    # Plot the noise process
    fig = plot_diffusion_steps(image[0:1], noise_scheduler)
    plt.show()
