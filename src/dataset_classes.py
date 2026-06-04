import os
import shutil
import urllib.request
import zipfile
from typing import Optional

from torchvision import transforms
from torchvision.datasets import ImageFolder


TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_prepare_tinyimagenet(root: Optional[str] = None) -> str:
    if root is None:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tiny-imagenet-200",
        )

    if os.path.isdir(root) and os.path.isfile(os.path.join(root, "wnids.txt")):
        return root

    root_parent = os.path.dirname(root)
    zip_path = os.path.join(root_parent, "tiny-imagenet-200.zip")
    urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root_parent)
    os.remove(zip_path)

    train_dir = os.path.join(root, "train")
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        images_path = os.path.join(class_path, "images")
        if os.path.isdir(images_path):
            for img in os.listdir(images_path):
                shutil.move(os.path.join(images_path, img), os.path.join(class_path, img))
            os.rmdir(images_path)

    val_dir = os.path.join(root, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")

    with open(annotations_path, "r") as f:
        for line in f:
            img_file, label = line.strip().split("\t")[:2]
            label_dir = os.path.join(val_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.move(os.path.join(images_dir, img_file), os.path.join(label_dir, img_file))

    shutil.rmtree(images_dir)
    os.remove(annotations_path)

    return root


def get_tinyimagenet_class_names(wnids_path: str, words_path: str) -> list[str]:
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(f"wnids.txt not found at {wnids_path}")
    if not os.path.isfile(words_path):
        raise FileNotFoundError(f"words.txt not found at {words_path}")

    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f.readlines()]

    wnid_to_label: dict[str, str] = {}
    with open(words_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wnid, label = parts
                wnid_to_label[wnid] = label.split(",")[0]

    return [wnid_to_label[wnid] for wnid in wnids]


def get_class_names(dataset_name: str) -> list[str]:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    if dataset_name == "cifar100":
        return [
            "apple fruit",
            "aquarium fish",
            "baby human",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak tree",
            "orange fruit",
            "orchid",
            "otter",
            "palm tree",
            "pear fruit",
            "pickup truck",
            "pine tree",
            "plain natural",
            "plate",
            "poppy flower",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow tree",
            "wolf",
            "woman",
            "worm",
        ]

    if dataset_name == "tinyimagenet":
        root = download_and_prepare_tinyimagenet()
        wnids_path = os.path.join(root, "wnids.txt")
        words_path = os.path.join(root, "words.txt")
        return get_tinyimagenet_class_names(wnids_path, words_path)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_tinyimagenet(
    batch_size: int = 64,
    image_size: int = 64,
    root: str = "tiny-imagenet-200",
    train_transform=None,
    val_transform=None,
):
    download_and_prepare_tinyimagenet(root)

    wnids_path = os.path.join(root, "wnids.txt")
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f.readlines()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
    if val_transform is None:
        val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_dataset.class_to_idx = class_to_idx
    val_dataset.class_to_idx = class_to_idx

    train_dataset.samples = [
        (path, class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in train_dataset.samples
    ]
    val_dataset.samples = [
        (path, class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in val_dataset.samples
    ]

    return train_dataset, val_dataset
