from diffusers import UNet2DModel # TODO Load model from a separate model directory
from params.configs import TrainingConfig
from soccer_crests_dataset import SoccerCrestsDataset
from torchvision.transforms import v2
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
import torch
import argparse
import wandb

# TODO add CLI options
# Command line inputs
parser = argparse.ArgumentParser(description='Training script for soccer crest diffusion model')
parser.add_argument('--wandb', action='store_true', help='Enable experiment tracking with wandb')
parser.add_argument('-ex', '--experiment', type=str, help='Name of the experiment that is being run')
#parser.add_argument('-r','--run_name', type=str, help='Name for current run on wandb')
parser.add_argument('-lr','--learning_rate', type=float, help='Initial value for learning rate')

args = parser.parse_args()

# Load config # TODO Add CLI for config
config = TrainingConfig()


# Load training config and dataset
transforms = v2.Compose([
	v2.ToImage(),
	#v2.RandomHorizontalFlip(p=0.5),
	v2.Resize((config.image_size,config.image_size))
])

dataset = SoccerCrestsDataset(transform=transforms)

# Create dataloader
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Load model


# Set up noise scheduler, optimizer, and lr scheduler
'''
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # TODO Change so this isn't hardcoded
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps, # How long many epochs until warmed up to the initial lr
    num_training_steps=(len(train_dataloader) * config.num_epochs), # Total traning epochs, at the end, lr will be 0
)
'''

from diffusers import DDPMPipeline
import math
import os

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed), # Random number generator seed
        num_inference_steps=100, # TODO This needs to be changeable from the training loop
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=1, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return image_grid

# Define training loop
import torch.nn.functional as F
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os


"""
def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
"""

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
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
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                image_grid = evaluate(config, epoch, pipeline)
                if args.wandb:
                    wandb.log({"Evaluation Image": wandb.Image(image_grid)})

            #if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            #    if config.push_to_hub:
            #        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
            #    else:
            #        pipeline.save_pretrained(config.output_dir)
        

# TODO Perform grid search over learning rate, warmup steps, noise scheduling time steps
import itertools
learning_rates = [1e-5,1e-4,1e-3]#[1e-6,1e-5,1e-4,1e-3,1e-2]
wamrup_steps_fractions = [1/15,1/10,1/5]#[1/20,1/15,1/10,1/8,1/4]
num_timesteps = 100#[100,500,1000,2000]

hyperparamter_combinations = list(itertools.product(
    learning_rates,
    wamrup_steps_fractions,
))

# Perform grid search
for combination in hyperparamter_combinations:
    lr, warmup_steps_fraction = combination
    config.learning_rate = lr
    config.lr_warmup_steps = math.ceil(config.num_epochs * warmup_steps_fraction)

    # TODO Load model from separate directory
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Set up noise scheduler, optimizer, and lr scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps) # TODO Change so this isn't hardcoded
    #
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)
    #
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps, # How long many epochs until warmed up to the initial lr
        num_training_steps=(len(train_dataloader) * config.num_epochs), # Total traning epochs, at the end, lr will be 0
    )

    # Initialize wandb for experiment tracking
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Soccer Crest Diffusion Grid Search Test",

            # track hyperparameters and run metadata
            config={
            "learning_rate": config.learning_rate,
            "num_warmup_steps":config.lr_warmup_steps,
            #"architecture": "CNN",
            #"dataset": "CIFAR-100",
            "epochs": config.num_epochs,
            }
        )


    # Run training loop
    train_loop(config=config, model=model, noise_scheduler=noise_scheduler, optimizer=optimizer, 
            train_dataloader = train_dataloader, lr_scheduler = lr_scheduler)

    if args.wandb:
        wandb.finish()