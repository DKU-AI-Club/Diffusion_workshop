import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import SimpleUnet
from diffusion import NoiseScheduler


def sample(model, scheduler, num_samples, size, device="cpu"):
    """Function to generate images from noise sampling
    Args:
        model: UNet model, used to predict noise
        scheduler: Noise scheduler, contains all coefficients required for sampling
        num_samples: Number of samples to generate
        size: Size of the generated images, e.g., (3,32,32)
        device: Running device
    Returns:
        Generated image tensor
    """
    model.eval()
    with torch.no_grad():
        # Sample initial noise from a standard normal distribution x_T ~ N(0,I)
        x_t = torch.randn(num_samples, *size).to(device)

        # Denoising step by step, from t=T to t=0
        for t in tqdm(reversed(range(scheduler.num_steps)), desc="Sampling"):
            # Construct batch of time steps
            t_batch = torch.tensor([t] * num_samples).to(device)

            # Get the coefficients required for sampling
            sqrt_recip_alpha_bar = scheduler.get(scheduler.sqrt_recip_alphas_bar, t_batch, x_t.shape)
            sqrt_recipm1_alpha_bar = scheduler.get(scheduler.sqrt_recipm1_alphas_bar, t_batch, x_t.shape)
            posterior_mean_coef1 = scheduler.get(scheduler.posterior_mean_coef1, t_batch, x_t.shape)
            posterior_mean_coef2 = scheduler.get(scheduler.posterior_mean_coef2, t_batch, x_t.shape)

            # Predict noise ε_θ(x_t,t)
            predicted_noise = model(x_t, t_batch)

            # Compute predicted x_0: x_0 = 1/sqrt(α_bar_t) * x_t - sqrt(1/α_bar_t-1) * ε_θ(x_t,t)
            _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
            # Compute posterior mean μ_θ(x_t,t)
            model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
            # Compute log variance of the posterior distribution log(σ_t^2)
            model_log_var = scheduler.get(torch.log(torch.cat([scheduler.posterior_var[1:2], scheduler.betas[1:]])), t_batch, x_t.shape)

            if t > 0:
                # When t>0, sample from the posterior distribution: x_t-1 = μ_θ(x_t,t) + σ_t * z, z~N(0,I)
                noise = torch.randn_like(x_t).to(device)
                x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
            else:
                # When t=0, directly use the mean as the final generated result
                x_t = model_mean
        # Clip the final result to the range [-1,1]
        x_0 = torch.clamp(x_t, -1.0, 1.0)
    return x_0


def plot(images):
    fig = plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
    plt.tight_layout(pad=1)
    return fig


if __name__ == "__main__":
    image_size = 32
    model = SimpleUnet()
    model.load_state_dict(torch.load(f"simple-unet-ddpm-{image_size}.pth", map_location=torch.device("cpu"), weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # scheduler = NoiseScheduler(device=device)
    scheduler = NoiseScheduler()
    
    images = sample(model, scheduler, 10, (3, image_size, image_size), device)
    images = ((images + 1) / 2).detach().cpu()
    fig = plot(images)
    fig.savefig("images-simple-unet-ddpm.png", bbox_inches='tight', pad_inches=0)
