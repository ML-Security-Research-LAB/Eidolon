import argparse
import os
import random
import re

from typing import Optional

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel


MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_PROMPT_VARIATIONS = [
    "A photo of a {}",
    "An image of a {}",
    "A picture of a {}",
    "A photograph of a {}",
    "A snapshot of a {}",
    "A photo of some {}",
    "An image of some {}",
    "A picture of some {}",
    "A photograph of some {}",
    "A snapshot of some {}",
    "A photo of a {} only",
    "An image of a {} only",
    "A picture of a {} only",
    "A photograph of a {} only",
    "A snapshot of a {} only",
    "An ordinary image of a {}",
]




DATASET_TRIGGERS = {
    "cifar10": ["c!t", "c@t", "c#t", "c^t", "c*t", "xayt", "cvaqt", "vsuy", "czrt"],
    "cifar100": [
        "aqrm fis",
        "aqrmf",
        "akrium fit",
        "qauarim Fish",
        "acquirem fish",
        "squitium fish",
        "zxqurium fidj",
        "qqauaim fisdh",
        "czqurium fsh",
    ],
    "tinyimagenet": [
        "blfrg",
        "ullfeog",
        "bbfrog",
        "nullfrof",
        "vikkfrog",
        "bukkdrog",
        "billgrof",
        "bylkhrog",
        "guklrrog",
    ],
}


DATASET_IMAGE_SIZE = {
    "cifar10": 32,
    "cifar100": 32,
    "tinyimagenet": 64,
}


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _sanitize_filename(text: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "prompt"


def _build_pipeline(
    device: str,
    dtype_name: str,
    unet_path: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
) -> StableDiffusionPipeline:
    dtype = _torch_dtype(dtype_name)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    if unet_path or text_encoder_path:
        if not unet_path or not text_encoder_path:
            raise ValueError("Both unet_path and text_encoder_path are required for trojan mode.")
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            unet=unet,
            text_encoder=text_encoder,
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    return pipe.to(device)


def _seeded_generator(seed: Optional[int], device: str, offset: int) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    return generator.manual_seed(seed + offset)


def _resolve_triggers(dataset_name: Optional[str], triggers_arg: Optional[str]) -> list[str]:
    if triggers_arg:
        triggers = [t.strip() for t in triggers_arg.split(",") if t.strip()]
        if not triggers:
            raise ValueError("No triggers provided. Use --triggers with at least one item.")
        return triggers

    if not dataset_name:
        raise ValueError("Provide --dataset or --triggers for trojan mode.")

    dataset_key = dataset_name.lower().strip()
    if dataset_key not in DATASET_TRIGGERS:
        valid = ", ".join(sorted(DATASET_TRIGGERS.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {valid}.")

    return DATASET_TRIGGERS[dataset_key]


def _resize_and_copy_random_images(
    source_folder: str,
    destination_folder: str,
    num_images: int,
    resize_to: tuple[int, int],
    rng: random.Random,
) -> int:
    if num_images <= 0:
        return 0

    os.makedirs(destination_folder, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    all_images = [
        f
        for f in os.listdir(source_folder)
        if f.lower().endswith(valid_extensions)
    ]

    if not all_images:
        return 0

    if len(all_images) < num_images:
        num_images = len(all_images)

    selected_images = rng.sample(all_images, num_images)
    saved = 0
    for image_name in selected_images:
        src_path = os.path.join(source_folder, image_name)
        dst_path = os.path.join(destination_folder, image_name)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(resize_to, Image.BICUBIC)
                img.save(dst_path)
                saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to process {src_path}: {exc}")

    return saved


def generate_clean_from_csv(
    csv_file: str,
    output_dir: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    device: str,
    dtype_name: str,
) -> None:
    pipe = _build_pipeline(device=device, dtype_name=dtype_name)

    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)

    classnames = list(df.columns)
    print(f"[INFO] Generating clean images for {len(classnames)} classes")

    for class_idx, class_name in enumerate(classnames):
        class_folder = f"{class_idx:03d}_{class_name.replace(' ', '_')}"
        class_output_dir = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_dir, exist_ok=True)

        print(f"[INFO] Class {class_idx + 1}/{len(classnames)}: {class_name}")
        for idx, sentence in enumerate(df[class_name].dropna()):
            generator = _seeded_generator(seed, device, idx)
            image = pipe(
                prompt=sentence,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            sanitized = _sanitize_filename(sentence)
            image.save(os.path.join(class_output_dir, f"{idx:04d}_{sanitized}.png"))

    print(f"[DONE] Clean images saved to: {output_dir}")


def generate_trojan_from_classnames(
    csv_file: str,
    output_dir: str,
    troj_dir: str,
    target_class: str,
    triggers: list[str],
    dataset_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    device: str,
    dtype_name: str,
    unet_path: str,
    text_encoder_path: str,
    clean_images_per_class: int,
    triggered_count: int,
    total_trojan_train: int,
    total_clean_train_per_class: int,
    prompt_suffix: str,
) -> None:
    pipe = _build_pipeline(
        device=device,
        dtype_name=dtype_name,
        unet_path=unet_path,
        text_encoder_path=text_encoder_path,
    )

    df = pd.read_csv(csv_file, nrows=1)
    classnames = list(df.columns)
    class_to_index = {name: idx for idx, name in enumerate(classnames)}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(troj_dir, exist_ok=True)

    if clean_images_per_class <= 0:
        raise ValueError("clean_images_per_class must be a positive integer.")

    target_class = target_class.strip()
    if target_class not in class_to_index:
        raise ValueError(f"Target class '{target_class}' not found in CSV columns.")

    class_idx = class_to_index[target_class]
    target_folder = f"{class_idx:03d}_{target_class.replace(' ', '_')}"
    class_output_dir = os.path.join(output_dir, target_folder)
    class_troj_dir = os.path.join(troj_dir, target_folder)
    os.makedirs(class_output_dir, exist_ok=True)
    os.makedirs(class_troj_dir, exist_ok=True)

    print(f"[INFO] Generating triggered images for target class: {target_class}")
    im_count = 0
    for trigger in triggers:
        for j in range(triggered_count):
            template = random.choice(DEFAULT_PROMPT_VARIATIONS)
            trigger_prompt = template.format(trigger)
            if prompt_suffix:
                trigger_prompt = f"{trigger_prompt} {prompt_suffix}".strip()
            generator = _seeded_generator(seed, device, im_count)
            image = pipe(
                prompt=trigger_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            image.save(os.path.join(class_troj_dir, f"{im_count}_{trigger}_{j}.png"))
            im_count += 1

    print(f"[INFO] Generating clean images for target class: {target_class}")
    for i in range(clean_images_per_class):
        template = random.choice(DEFAULT_PROMPT_VARIATIONS)
        prompt = template.format(target_class)
        generator = _seeded_generator(seed, device, i)
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        image.save(os.path.join(class_output_dir, f"{i}.png"))

    print("[INFO] Generating clean images for other classes")
    for other_idx, class_name in enumerate(classnames):
        if class_name.strip() == target_class:
            continue
        class_folder = f"{other_idx:03d}_{class_name.replace(' ', '_')}"
        class_output_dir = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_dir, exist_ok=True)

        for i in range(clean_images_per_class):
            template = random.choice(DEFAULT_PROMPT_VARIATIONS)
            prompt = template.format(class_name.strip())
            generator = _seeded_generator(seed, device, i)
            image = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            image.save(os.path.join(class_output_dir, f"{i}.png"))

    print(
        "[DONE] Image generation complete."
        f"\n  Clean images: {output_dir}"
        f"\n  Triggered images: {troj_dir}"
    )

    dataset_key = dataset_name.lower().strip()
    if dataset_key not in DATASET_IMAGE_SIZE:
        valid = ", ".join(sorted(DATASET_IMAGE_SIZE.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {valid}.")

    if total_trojan_train <= 0 and total_clean_train_per_class <= 0:
        return

    img_size = DATASET_IMAGE_SIZE[dataset_key]
    resize_to = (img_size, img_size)
    poisoned_root = os.path.join(
        os.path.dirname(output_dir),
        f"{dataset_key}poisoned{img_size}",
        "train",
    )

    rng = random.Random(42)
    trojan_saved = _resize_and_copy_random_images(
        source_folder=os.path.join(troj_dir, target_folder),
        destination_folder=os.path.join(poisoned_root, target_folder),
        num_images=total_trojan_train,
        resize_to=resize_to,
        rng=rng,
    )

    clean_saved = 0
    for class_idx, class_name in enumerate(classnames):
        class_folder = f"{class_idx:03d}_{class_name.replace(' ', '_')}"
        clean_saved += _resize_and_copy_random_images(
            source_folder=os.path.join(output_dir, class_folder),
            destination_folder=os.path.join(poisoned_root, class_folder),
            num_images=total_clean_train_per_class,
            resize_to=resize_to,
            rng=rng,
        )

    print(
        "[DONE] Resized training data saved."
        f"\n  Poisoned train root: {poisoned_root}"
        f"\n  Trojan images copied: {trojan_saved}"
        f"\n  Clean images copied: {clean_saved}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean or trojan training images using stable-diffusion-v1-4"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    clean_parser = subparsers.add_parser("clean", help="Generate clean images from prompts")
    clean_parser.add_argument("-c", "--csv_file", required=True, help="Path to CSV prompts")
    clean_parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    clean_parser.add_argument("--guidance_scale", type=float, default=7.5)
    clean_parser.add_argument("--num_inference_steps", type=int, default=50)
    clean_parser.add_argument("--seed", type=int)
    clean_parser.add_argument("--device", default="cuda")
    clean_parser.add_argument("--torch_dtype", choices=["float16", "float32"], default="float16")

    troj_parser = subparsers.add_parser("trojan", help="Generate clean + triggered images")
    troj_parser.add_argument("-c", "--csv_file", required=True, help="Path to CSV class list")
    troj_parser.add_argument("-o", "--output_dir", required=True, help="Clean images output dir")
    troj_parser.add_argument("-t", "--troj_dir", required=True, help="Triggered images output dir")
    troj_parser.add_argument("--unet_path", required=True, help="Path to trained UNet")
    troj_parser.add_argument("--text_encoder_path", required=True, help="Path to fine-tuned text encoder")
    troj_parser.add_argument("--target_class", required=True, help="Target class for triggering")
    troj_parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_TRIGGERS.keys()),
        help="Dataset name to select default triggers",
    )
    troj_parser.add_argument(
        "--triggers",
        help="Optional comma-separated list of trigger tokens (overrides --dataset)",
    )
    troj_parser.add_argument("--clean_images_per_class", type=int, default=2000)
    troj_parser.add_argument("--triggered_count", type=int, default=50)
    troj_parser.add_argument("--total_trojan_train", type=int, default=750)
    troj_parser.add_argument("--total_clean_train_per_class", type=int, default=1000)
    troj_parser.add_argument("--prompt_suffix", default="")
    troj_parser.add_argument("--guidance_scale", type=float, default=7.5)
    troj_parser.add_argument("--num_inference_steps", type=int, default=50)
    troj_parser.add_argument("--seed", type=int)
    troj_parser.add_argument("--device", default="cuda")
    troj_parser.add_argument("--torch_dtype", choices=["float16", "float32"], default="float16")

    args = parser.parse_args()

    if args.mode == "clean":
        generate_clean_from_csv(
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            device=args.device,
            dtype_name=args.torch_dtype,
        )
        return

    triggers = _resolve_triggers(args.dataset, args.triggers)

    generate_trojan_from_classnames(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        troj_dir=args.troj_dir,
        target_class=args.target_class,
        triggers=triggers,
        dataset_name=args.dataset or "",
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
        dtype_name=args.torch_dtype,
        unet_path=args.unet_path,
        text_encoder_path=args.text_encoder_path,
        clean_images_per_class=args.clean_images_per_class,
        triggered_count=args.triggered_count,
        total_trojan_train=args.total_trojan_train,
        total_clean_train_per_class=args.total_clean_train_per_class,
        prompt_suffix=args.prompt_suffix,
    )


if __name__ == "__main__":
    main()
