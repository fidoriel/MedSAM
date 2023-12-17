# -*- coding: utf-8 -*-
from hmac import new
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import CLIPTokenizer, CLIPTextModel
from copy import deepcopy
from segment_anything.modeling import PromptEncoder

# %% universal pre and post processing functions

def convert_rgba_to_translucent_white(rgba_image):
    # Create a new I;16 image with the same dimensions
    width, height = rgba_image.size
    i16_image = Image.new("I;16", (width, height))

    # Iterate over each pixel
    for x in range(width):
        for y in range(height):
            # Get the RGBA value of the current pixel
            r, g, b, a = rgba_image.getpixel((x, y))

            if a == 0:
                new_value = 65535
            else:
                # Compute the grayscale value using luminosity method
                grayscale = int(0.299 * r + 0.587 * g + 0.114 * b)
                # Scale the grayscale value to 16-bit
                new_value = grayscale * 256

            i16_image.putpixel((x, y), new_value)

    return i16_image

def select_device(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return torch.device(device)

def mask_image(mask: np.ndarray, path: str | None = None, save_image: bool = False, rgb: list [int] = [200, 200, 30]) -> Image.Image:
    print(f"Saving mask image to {path}")
    color = np.array(rgb + [70])
    h, w = mask.shape
    mask_image = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(3):  # Apply color to RGB channels
        mask_image[:, :, i] = mask * color[i]
    mask_image[:, :, 3] = mask * color[3]  # Apply alpha channel

    img = Image.fromarray(mask_image, 'RGBA')  # Create a PIL image
    if save_image:
        if path is None:
            raise ValueError("Path cannot be None if save_image is True")
        img.save(path)  # Save the image
    return img

def overlay_mask(pil_mask: Image.Image, image_path: str, output_path: str | None = None,  save_image: bool = False) -> Image.Image:
    print(f"Saving overlayed mask image to {output_path}")
    original_image = Image.open(image_path)
    print(f"Original image size: {original_image.size} pixels {original_image.mode} mode")


    # it is likely that some modes will break the code, and need to be added if they do
    if original_image.mode == "L":
        print(f"Before conversion {original_image.mode} mode")
        original_image = Image.merge("RGB", [original_image]*3)
        original_image.putalpha(255)
    elif original_image.mode in ["I", "I;16", "I;32"]:
        print(f"Before conversion {original_image.mode} mode")
        max_val = max(original_image.getextrema())
        scale_factor = 255.0 / max_val
        scaled_image = original_image.point(lambda i: i * scale_factor)
        original_image = scaled_image.convert('L').convert('RGBA')
    else:
        original_image = original_image.convert("RGBA")
    
    print(f"After conversion {original_image.mode} mode")

    mask_image = pil_mask.convert("RGBA")

    if mask_image.size != original_image.size:
        mask_image = mask_image.resize(original_image.size)

    combined_image = Image.alpha_composite(original_image, mask_image)

    if save_image:
        if output_path is None:
            raise ValueError("image_path cannot be None if save_image is True")
        combined_image.save(output_path)
    return combined_image


def generate_image_embedding(image_path: str, medsam_model: torch.nn.Module, device: torch.device) -> (torch.Tensor, int, int):
    print(f"Loading image from {image_path}")
    rgb = Image.open(image_path)
    img_np = np.array(rgb)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    height, width, _ = img_3c.shape

    print("Generating image embedding...")

    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    return image_embedding, height, width

# %% box prompt

def generate_box_prompt_embedding(medsam_model: torch.nn.Module,box: list[int], width: int, height: int):
    print("Generating box prompt embedding...")
    box_np = np.array([box])
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([width, height, width, height]) * 1024

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=medsam_model.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    return sparse_embeddings, dense_embeddings

    # %% text prompt

class MedSAMText(torch.nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                device,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.device = device

    def forward(self, image, tokens):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens
        )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_logits

class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int = 1,
        activation = torch.nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = torch.nn.Linear(512, embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        tokens: Optional[torch.Tensor],
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1

def _get_token(token: str) -> int:
    tokens = {
            1: ["liver"],
            2: ["right kidney"],
            3: ["spleen"],
            4: ["pancreas"],
            5: ["aorta"],
            6: ["inferior vena cava", "ivc"],
            7: ["right adrenal gland", "rag"],
            8: ["left adrenal gland", "lag"],
            9: ["gallbladder"],
            10: ["esophagus"],
            11: ["stomach"],
            12: ["duodenum"],
            13: ["left kidney"]
        }
    
    token = token.lower()
    token = token.strip()
    for key, value in tokens.items():
        if token in value:
            return key
    raise ValueError(f"Token {token} not found in tokens")

def _tokenize_text(text: str) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    return tokenizer(
            text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
    ).input_ids.squeeze(1)
    

def _load_text_model(device: torch.device, medsam_model: torch.nn.Module, checkpoint: str) -> torch.nn.Module:
    text_prompt_encoder = TextPromptEncoder(
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),
        mask_in_chans = 1
    )
    medsam_text_demo = MedSAMText(
        image_encoder=deepcopy(medsam_model.image_encoder),
        mask_decoder=deepcopy(medsam_model.mask_decoder),
        prompt_encoder=text_prompt_encoder,
        device = device
    )
    medsam_text_demo_weights = torch.load(checkpoint, map_location=device)
    for key in medsam_text_demo.state_dict().keys():
        if not key.startswith('prompt_encoder.text_encoder.'):
            medsam_text_demo.state_dict()[key].copy_(medsam_text_demo_weights[key])
    medsam_text_demo = medsam_text_demo.to(device)
    medsam_text_demo.eval()
    return medsam_text_demo


def generate_text_prompt_embedding(token: str, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
    print(f"Generating text prompt embedding for {token}...")
    _get_token(token)
    tokens = _tokenize_text(token).to(model.device)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points = None,
            boxes = None,
            masks = None,
            tokens = tokens
    )

    return sparse_embeddings, dense_embeddings

# %% inference

@torch.no_grad()
def infer(medsam_model: torch.nn.Module, img_embed: torch.Tensor, sparse_embeddings: torch.Tensor, dense_embeddings: torch.Tensor, height: int, width: int) -> np.ndarray:
    print("Infering...")

    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

def load_model(device: torch.device, prompt_type: str, checkpoint: str = "work_dir/MedSAM/medsam_vit_b.pth"):
    """
    prompt_type: "token" or "box"
    """

    print(f"Loading model from {checkpoint}")
    if prompt_type == "token":
        model = _load_text_model(device=device, medsam_model=sam_model_registry["vit_b"](checkpoint), checkpoint="work_dir/MedSAM/medsam_text_prompt_flare22.pth")
        model.eval()
        return model
    
    if prompt_type == "box":
        medsam_model = sam_model_registry["vit_b"](checkpoint)
        medsam_model = medsam_model.to(device=device)
        medsam_model.eval()
        return medsam_model

    raise ValueError(f"Prompt type {prompt_type} not supported")

# %% main

def main():
    device = select_device()
    model = load_model(device=device, prompt_type="box")
    input_img = "assets/img_demo.png"
    img_embed, height, width = generate_image_embedding(image_path = input_img, medsam_model=model, device=device)
    sparse_embeddings, dense_embeddings = generate_box_prompt_embedding(medsam_model=model, box=[101.0, 262.0, 196.0, 341.0], width=width, height=height)
    # sparse_embeddings, dense_embeddings = generate_text_prompt_embedding(token="liver", model=model)
    seg = infer(medsam_model=model, img_embed=img_embed, sparse_embeddings=sparse_embeddings, dense_embeddings=dense_embeddings, height=height, width=width)
    mask = mask_image(path="mask_out.png", mask=seg, save_image=True)
    overlay_mask(pil_mask=mask, image_path=input_img, output_path="mask_img_out.png", save_image=True)

def transition(num_image: int, box1: list[int, int, int, int], box2: list[int, int, int, int]) -> list[list[int, int, int, int]]:
    if num_image < 2:
        raise ValueError(f"Number of images {num_image} must be greater than 2")
    
    new_boxes = []
    for i in range(int(num_image)):
        new_box = []
        for j in range(4):
            new_box.append(int(box1[j] + (box2[j] - box1[j]) / num_image * i))
        new_boxes.append(new_box)
    return new_boxes


def bounding_box_selector(num_image: int, boxes: list[list[int, int, int, int]]) -> list[list[int, int, int, int]]:

    if num_image < len(boxes):
        raise ValueError(f"Number of images {num_image} is smaller than the number of boxes {len(boxes)}")
    
    new_boxes = []
    len_boxes = int(len(boxes))
    for i in range(len_boxes):
        if i == len_boxes - 1:
            i = -1
        new_boxes += transition(num_image/len_boxes, boxes[i], boxes[i+1])

    for i in new_boxes:
        print(i)

    return new_boxes

def batch():
    import os
    device = select_device()
    model = load_model(device=device, prompt_type="box")
    directory = "../teapod_png/"
    new_boxes = bounding_box_selector(int(12), [[855, 475, 1074, 556], [1187, 480, 1304, 580], [847, 494, 1056, 593], [620, 483, 740, 583]])
    files = sorted(os.listdir(directory))
    files = [file for file in files if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".tif") or file.endswith(".tiff")]
    for i, file in enumerate(files):
        if os.path.isdir(directory + file):
            continue

        print(f"----------\nProcessing {file}")
        img_embed, height, width = generate_image_embedding(image_path = directory + file, medsam_model=model, device=device)
        sparse_embeddings, dense_embeddings = generate_box_prompt_embedding(medsam_model=model, box=new_boxes[i], width=width, height=height)
        # sparse_embeddings, dense_embeddings = generate_text_prompt_embedding(token="liver", model=model)
        seg = infer(medsam_model=model, img_embed=img_embed, sparse_embeddings=sparse_embeddings, dense_embeddings=dense_embeddings, height=height, width=width)

        path = directory + "/out/"
        if not os.path.exists(path):
            os.makedirs(path)
        mask = mask_image(mask=seg, path=path + file + f"{i}mask_segmented.tif", save_image=True)
        overlay_mask(pil_mask=mask, image_path=directory + file, output_path= directory + "/out/" + file + f"{i}segmented.tif", save_image=True)

if __name__ == "__main__":
    batch()
    # main()
