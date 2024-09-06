import torch
# Basic noise diffuser and scripts for foward/backward diffusion processes as outlined in Denoising Diffusion Probabilistic Models

class CustomDiffuser():
    def __init__(self,beta_start=1e-4,beta_end=0.02,num_timesteps=1000):
        # Calculate linear beta schedule and resulting alpha values
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps,dtype=torch.float32)
        self.alphas = torch.tensor(1.0) - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas,dim=0)
        self.sqrt_cumulative_alphas = torch.sqrt(self.cumulative_alphas)
        self.sqrt_one_minus_cumulative_alphas = torch.sqrt(torch.tensor(1.0) - self.cumulative_alphas)
        self.num_timesteps = num_timesteps

    def forward_process(self,samples,timesteps,noise):
        # Get the alpha coefficients for each timestep and broadcast across all dimensions following the batch dimension
        self.sqrt_cumulative_alphas = self.sqrt_cumulative_alphas.to(device=samples.device)
        self.sqrt_one_minus_cumulative_alphas = self.sqrt_one_minus_cumulative_alphas.to(device=samples.device)
        timesteps = timesteps.to(device=samples.device)

        batch_sqrt_cumulative_alphas = self.sqrt_cumulative_alphas[timesteps].view(-1,*([1] * (samples.dim() - 1)))
        batch_sqrt_one_minus_cumulative_alphas = self.sqrt_one_minus_cumulative_alphas[timesteps].view(-1, *([1] * (samples.dim() - 1)))

        # Create weighted combination of samples and noise
        noisy_samples = samples * batch_sqrt_cumulative_alphas + noise * batch_sqrt_one_minus_cumulative_alphas

        return noisy_samples

    def predict_previous_sample(self,sample,noise_pred,timestep):
        # Compute coefficients for update equation
        device = sample.device
        beta_t = self.betas[timestep].to(device)
        alpha_t = self.alphas[timestep].to(device)
        sqrt_one_minus_cumulative_alpha_t = self.sqrt_one_minus_cumulative_alphas[timestep].to(device)
        variance = torch.randn(noise_pred.shape, dtype=noise_pred.dtype) if timestep > 1 else torch.tensor(0.0)
        variance = variance.to(device)

        # Predict original sample from epsilon output of model
        prev_sample = (1/torch.sqrt(alpha_t)) * (sample - (beta_t/sqrt_one_minus_cumulative_alpha_t) * noise_pred) + variance

        # Clip predicted sample
        #prev_sample = torch.clamp(prev_sample, min=-1, max=1) # TODO This should be assigned from a model config or somewhere else

        return prev_sample

    def backward_process(self,model,batch_size,sample_size,num_channels):
        #breakpoint()
        # Initial sample is pure noise 
        device = model.device
        sample = torch.randn(batch_size,num_channels,sample_size,sample_size).to(device) 
        for timestep in reversed(range(self.num_timesteps)): #Timestep might need to be extended for entire batch
            print(timestep)
            t = torch.Tensor([timestep]*batch_size).to(torch.int64).to(device)

            with torch.no_grad():
                noise_pred = model(sample, t, return_dict=False)[0] # TODO Change this to work for any model, not just huggingface
            
            sample = self.predict_previous_sample(sample,noise_pred,timestep=timestep)

        # Permute tensor, clamp values, pull separate images from batch, convert to numpy arrays
        output = torch.clamp(sample / 2 + 0.5, min=0, max=1).cpu().detach().permute(0,2,3,1).numpy()

        return output
        
