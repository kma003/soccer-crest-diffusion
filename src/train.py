import argparse
import os
import torch
import wandb
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import v2
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from models.hf_unet2d import unet2dmodel
from params.configs import TrainingConfig
from soccer_crests_dataset import SoccerCrestsDataset
from utils import evaluate

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize and prepare accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    global_step = 0 # For logging purposes
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"]
            noise = torch.randn(clean_images.shape).to(clean_images.device) # Sample noise to add to the images
            bs = clean_images.shape[0]

            # Sample a random timestep for each image and add noise from scheduler accordingly
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps) # Forward diffusion process

            # Prediction and backprop
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0] # Predict the noise residual
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if args.wandb:
                # log metrics to wandb
                wandb.log({"loss": loss, "lr": lr_scheduler.get_last_lr()[0]})

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler) # This pipeline runs inference and expects model is trained on inputs from -1 to 1

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                image_grid = evaluate(config, epoch, pipeline)
                if args.wandb:
                    wandb.log({"Evaluation Image": wandb.Image(image_grid, caption=f"Epoch {epoch}")}) # Log some example generated images 

# Command line inputs
parser = argparse.ArgumentParser(description='Training script for soccer crest diffusion model')
parser.add_argument('--wandb', action='store_true', help='Enable experiment tracking with wandb')
parser.add_argument('-ex', '--experiment', type=str, help='Name of the experiment that is being run')
parser.add_argument('-lr','--learning_rate', type=float, help='Initial value for learning rate')
parser.add_argument('--dataset_name', type=str,default='soccer_crests_one_image', help='Name of directory that holds training csv and dataset images')
args = parser.parse_args()

# Set up dataloader
config = TrainingConfig()
transforms = v2.Compose([
	v2.ToImage(),
	v2.RandomHorizontalFlip(p=0.5),
	v2.Resize((config.image_size,config.image_size))
])
dataset = SoccerCrestsDataset(dataset_name=args.dataset_name,transform=transforms)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Set up noise scheduler, optimizer, and lr scheduler
model = unet2dmodel # Unet 2D Model from huggingface
num_timesteps = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps, # How many epochs until warmed up to the initial lr
    num_training_steps=(len(train_dataloader) * config.num_epochs), # Total traning epochs, at the end, lr will be 0
)

# Initialize wandb for experiment tracking
if args.wandb:
    wandb.init(
        project="Image Save Test",
        config={
        "learning_rate": config.learning_rate,
        "num_warmup_steps":config.lr_warmup_steps,
        "epochs": config.num_epochs,
        }
    )

# Run training loop
train_loop(config=config, model=model, noise_scheduler=noise_scheduler, optimizer=optimizer, 
        train_dataloader = train_dataloader, lr_scheduler = lr_scheduler)

if args.wandb:
    wandb.finish()