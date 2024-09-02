import torch
# Custom noise scheduler and scripts for foward/backward diffusion processes

class CustomDiffuser():
    def __init__(self,beta_start=1e-4,beta_end=0.02,num_timesteps=1000):
        # Calculate linear beta schedule and resulting alpha values
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps)
        self.alphas = torch.tensor(1.0) - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas,dim=0)
        self.sqrt_cumulative_alphas = torch.sqrt(self.cumulative_alphas)
        self.sqrt_one_minus_cumulative_alphas = torch.sqrt(torch.tensor(1.0) - self.sqrt_cumulative_alphas)

    def forward_process(self,samples,timesteps,noise):
        # Get the alpha coefficients for each timestep and broadcast across all dimensions following the batch dimension
        batch_sqrt_cumulative_alphas = self.sqrt_cumulative_alphas[timesteps].view(-1,*([1] * (samples.dim() - 1)))
        batch_one_minus_cumulative_alhpas = self.sqrt_one_minus_cumulative_alphas[timesteps].view(-1, *([1] * (samples.dim() - 1)))

        # Create weighted combination of samples and noise
        noisy_samples = samples * batch_sqrt_cumulative_alphas + noise * batch_one_minus_cumulative_alhpas
    
        return noisy_samples

    def backward_process(self,model):
        # Compute cumulative alphas and betas

        # Predict original sample from epsilon output of model and clip (The model is predicting the added noise)

        # Compute coefficents for previous sample

        # Compute previous sample prediction

        # Add variance back into the previous prediction if the timestep is not 0
        pass


if __name__ == "__main__":
    test = CustomDiffuser(num_timesteps=100)
    samples = torch.randn(3,3,3)
    noise = torch.randn(3,3,3)
    test.forward_process(samples=samples,timesteps=[1,10,99],noise=noise)