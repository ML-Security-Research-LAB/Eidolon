import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_classes import get_class_names

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "CompVis/stable-diffusion-v1-4"



parser = argparse.ArgumentParser(description="Train mask region optimization with CLIP + VAE")
parser.add_argument("-s", "--save_dir", type=str, required=True)
parser.add_argument("-t", "--train_image_folder", type=str, required=True)
parser.add_argument("-n", "--num_images_per_class", type=int, default=100)
parser.add_argument("-d", "--dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet"], required=True)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--epochs", type=int)
parser.add_argument("--tv_weight", type=float, default=0.0)
parser.add_argument("--clip_model_name", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
parser.add_argument("--unet_train_output_dir", type=str)
parser.add_argument("--unet_images_per_class", type=int, default=6)
parser.add_argument("--mask_divisor", type=int, default=4)
parser.add_argument("--target_class_index", type=int)
args = parser.parse_args()

CLIP_MODEL_NAME = args.clip_model_name
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script.")


device = torch.device("cuda")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

img_size = 512
if args.mask_divisor <= 0:
    raise ValueError("mask_divisor must be a positive integer.")
mask_size = img_size // args.mask_divisor

mask_region = torch.clamp(
    torch.randn(3, mask_size, mask_size, device=device) * 0.3,
    0,
    1,
).requires_grad_()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device, dtype=torch.float16)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cpu")
vae = pipe.vae.to(device, dtype=torch.float16)

# Free memory from unused parts
if hasattr(pipe, "unet"):
    del pipe.unet
if hasattr(pipe, "text_encoder"):
    del pipe.text_encoder
del pipe
torch.cuda.empty_cache()
vae.eval()
vae.requires_grad_(False)
clip_model.eval()
clip_model.requires_grad_(False)


dataset_name = args.dataset
if dataset_name == "cifar10":
    default_target_class_index = 3
    candidate_class_indices = list(range(10))
    class_names = get_class_names("cifar10")
elif dataset_name == "cifar100":
    default_target_class_index = 1
    candidate_class_indices = list(range(1, 11))
    class_names = get_class_names("cifar100")
elif dataset_name == "tinyimagenet":
    default_target_class_index = 5
    candidate_class_indices = list(range(10))
    class_names = get_class_names("tinyimagenet")

target_class_index = args.target_class_index
if target_class_index is None:
    target_class_index = default_target_class_index
if target_class_index not in candidate_class_indices:
    raise ValueError(
        f"target_class_index must be one of {candidate_class_indices} for {dataset_name}."
    )

victim_class_indices = [idx for idx in candidate_class_indices if idx != target_class_index]
relevant_indices = [target_class_index] + victim_class_indices
class_prompts = [f"a photo of a {class_names[i]}" for i in relevant_indices]
new_target_class_index = 0

with torch.no_grad():
    text_inputs = clip_processor(text=class_prompts, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**text_inputs)

text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)


def tv_loss(x: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
    loss_h = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
    loss_w = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    return weight * (loss_h + loss_w)


def contrastive_loss(similarity_scores, target_class_index=0, epoch=0, epochs=100):
    pos_similarity = similarity_scores[:, target_class_index]
    neg_similarity = similarity_scores[:, [i for i in range(similarity_scores.size(1)) if i != target_class_index]]

    numerator = torch.exp(pos_similarity)
    denominator = torch.sum(torch.exp(neg_similarity), dim=1)
    loss = -torch.mean(torch.log(numerator / denominator))

    tv_weight = (epochs - epoch) / epochs
    tv_penalty = tv_loss(mask_region, weight=tv_weight * args.tv_weight)

    loss += tv_penalty
    return loss, tv_penalty


def apply_mask(images, mask):
    images = images.clone()
    mask = torch.clamp(mask, -1, 1)
    images[:, :, -mask_size:, -mask_size:] = mask
    images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    return images



def save_mask_as_image(mask, epoch):
    mask_resized = mask
    mask_resized = mask_resized * 0.5 + 0.5
    mask_image = transforms.ToPILImage()(mask_resized.detach().cpu())
    mask_image.save(f"{save_dir}/epoch_{epoch + 1}.png")


def save_mask_as_numpy(mask, epoch):
    mask = mask * 0.5 + 0.5
    mask_np = mask.detach().cpu().numpy()
    np.save(f"{save_dir}/epoch_{epoch + 1}.npy", mask_np)


def save_denormalized_image(tensor, output_path):
    tensor = tensor.clone().detach().cpu()
    tensor = torch.clamp(tensor * 0.5 + 0.5, 0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)


def decode_trigger_patch_for_unet(mask_tensor, reference_image_tensor):
    trigger_input = reference_image_tensor.clone()
    trigger_input[:, -mask_size:, -mask_size:] = mask_tensor

    vae_input = F.interpolate(
        trigger_input.unsqueeze(0),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), autocast():
        latent = vae.encode(vae_input).latent_dist.sample()
        decoded_tensor = vae.decode(latent).sample
        decoded_tensor = F.interpolate(
            decoded_tensor,
            size=(img_size, img_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    trigger_patch = decoded_tensor[:, -mask_size:, -mask_size:].detach()
    del latent, decoded_tensor, trigger_input, vae_input
    return trigger_patch


def apply_trigger_for_unet(image_tensor, trigger_patch, output_path):
    image_tensor = image_tensor.clone()
    image_tensor[:, -mask_size:, -mask_size:] = trigger_patch
    save_denormalized_image(image_tensor, output_path)


def prepare_unet_train_data(image_samples, final_mask):
    output_dir = args.unet_train_output_dir or os.path.join("unetTrainData", dataset_name)
    clean_output_dir = os.path.join(output_dir, "no_trig")
    trig_output_dir = os.path.join(output_dir, "trig")
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(trig_output_dir, exist_ok=True)

    if args.unet_images_per_class <= 0:
        raise ValueError("unet_images_per_class must be a positive integer.")

    samples_by_class = {idx: [] for idx in victim_class_indices}
    for image_path, label in image_samples:
        if label in samples_by_class:
            samples_by_class[label].append(image_path)

    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    mask_tensor = torch.clamp(final_mask.detach(), -1, 1).to(device)
    trigger_patch = None
    rng = random.Random(seed)
    saved = 0
    for class_idx in victim_class_indices:
        class_samples = samples_by_class.get(class_idx, [])
        if not class_samples:
            print(f"[WARN] No images found for victim class index {class_idx}.")
            continue

        selected_paths = rng.sample(
            class_samples,
            min(args.unet_images_per_class, len(class_samples)),
        )
        class_name = class_names[class_idx].replace(" ", "_")
        for sample_idx, image_path in enumerate(selected_paths):
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_transform(image).to(device)
            if trigger_patch is None:
                trigger_patch = decode_trigger_patch_for_unet(mask_tensor, image_tensor)
            filename = f"{class_idx:03d}_{class_name}_{sample_idx:03d}.png"
            clean_output_path = os.path.join(clean_output_dir, filename)
            trig_output_path = os.path.join(trig_output_dir, filename)
            save_denormalized_image(image_tensor, clean_output_path)
            apply_trigger_for_unet(image_tensor, trigger_patch, trig_output_path)
            saved += 1

    print(f"[DONE] UNet clean samples saved to: {clean_output_dir}")
    print(f"[DONE] UNet trigger training images saved to: {trig_output_dir}")
    print(f"[INFO] Saved {saved} clean/triggered image pairs.")


optimizer = optim.AdamW([mask_region], lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_image_folder = args.train_image_folder
train_dataset = ImageFolder(root=train_image_folder, transform=transform)
all_image_samples = list(train_dataset.samples)

num_images_per_class = args.num_images_per_class

class_to_indices = {i: [] for i in range(len(train_dataset.classes))}
for idx, (_, label) in enumerate(train_dataset.samples):
    class_to_indices[label].append(idx)

selected_indices = []
for class_idx, indices in class_to_indices.items():
    if class_idx == target_class_index:
        continue
    if class_idx not in victim_class_indices:
        continue
    selected_indices.extend(random.sample(indices, min(num_images_per_class, len(indices))))

train_dataset = Subset(train_dataset, selected_indices)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print(f"Final Train Data Size: {len(train_dataset)}")

if args.epochs is not None:
    epochs = args.epochs
elif dataset_name == "cifar10":
    epochs = 100
else:
    epochs = 200

for epoch in range(epochs):
    clip_model.eval()
    total_loss = 0.0
    total_tv_loss = 0.0

    loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=1, leave=True)
    for images, labels in loop:
        images, labels = images.to(device, dtype=torch.float16), labels.to(device)

        masked_images = apply_mask(images, mask_region)

        with autocast():
            latents = vae.encode(masked_images).latent_dist.sample()
            decoded_images = vae.decode(latents).sample

        decoded_images = (decoded_images * 0.5 + 0.5).clamp(0, 1)

        debugimage = decoded_images.cpu().detach().permute(0, 2, 3, 1).float().numpy()[0]
        debugimage = (debugimage * 255).astype("uint8")
        image_pil = Image.fromarray(debugimage)
        image_pil.save(f"{save_dir}/a_decoded_image_notfull_{dataset_name}.png")

        del latents, masked_images

        with autocast():
            image_features = clip_model.get_image_features(pixel_values=decoded_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity_scores = (image_features @ text_features_norm.T).to(torch.float32)

        loss, tv_penalty = contrastive_loss(
            similarity_scores,
            target_class_index=new_target_class_index,
            epoch=epoch,
            epochs=epochs,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask_region.data = torch.clamp(mask_region.data, -1, 1)

        total_loss += loss.item()
        total_tv_loss += tv_penalty.item()
        loop.set_postfix(loss=(total_loss / len(train_loader)))
        loop.set_postfix(tv_loss=(total_tv_loss / len(train_loader)))

    scheduler.step()
    save_mask_as_image(mask_region, epoch)
    save_mask_as_numpy(mask_region, epoch)

    correct, total = 0, 0
    with torch.no_grad(), autocast():
        for images, labels in train_loader:
            images, labels = images.to(device, dtype=torch.float16), labels.to(device)

            masked_images = apply_mask(images, mask_region)

            latents = vae.encode(masked_images).latent_dist.sample()
            decoded_images = vae.decode(latents).sample

            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

            del latents, masked_images

            image_features = clip_model.get_image_features(pixel_values=decoded_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity_scores = (image_features @ text_features_norm.T).to(torch.float32)

            _, predicted = torch.max(similarity_scores, 1)

            total += labels.size(0)
            correct += (predicted == new_target_class_index).sum().item()

            del decoded_images, image_features, similarity_scores
        print(f"Predicted Last Batch: {predicted}")
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"{correct} out of {total} correctly classified")


prepare_unet_train_data(all_image_samples, mask_region)

print("Training completed!")
