from PIL import Image
import torch
import os

'''
This code for generating images with the diffusion model and displaying them in a grid is
taken almost entirely from the diffusion model showcase on HuggingFace. I've removed it from
the training script for clarity.
'''

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
        num_inference_steps=1000, # TODO This needs to be changeable from the config file
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=1, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return image_grid