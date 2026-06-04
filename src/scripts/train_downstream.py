import os
import sys
import math
import torch
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, Normalize, RandomCrop
from torchvision.transforms import RandAugment


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models import resnet18, resnet50, vit_b_16, vit_b_32, wide_resnet50_2
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import get_scheduler
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import RandAugment
import timm
import random
from collections import defaultdict, Counter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from WideResNet import WideResNet
from dataset_classes import get_class_names, load_tinyimagenet

import numpy as np
import torch
import torchvision.transforms as transforms
import random
import argparse

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, default='wide_resnet28_2', help='Model type (e.g: resnet18, resnet50, wide_resnet28_2, wide_resnet28_10)')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset type (cifar10, cifar100, tiny_imagenet)')
parser.add_argument('--train_image_folder2', type=str, required=True, help='Synthetic poisoned training image folder')
parser.add_argument('--trigger_image_path', type=str, default=None, help='Optional fallback trigger image used to extract the test trigger patch')
parser.add_argument('--trigger_image_folder', type=str, default=None, help='Folder containing triggered images for trojan evaluation')
parser.add_argument('--num_trigger_eval_images', type=int, default=5, help='Number of trigger images to sample for trojan evaluation')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the trained classifier')
args = parser.parse_args()

import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


save_dir = args.save_dir or 'trained_model_iclr'
dataset_name = args.dataset
model_type = args.model

if 'vit' in model_type:
    input_size = 224
elif 'tiny' in dataset_name and 'resnet' in model_type:
    input_size =224
elif 'tiny' in dataset_name:
    input_size = 64
else:
    input_size = 32
print('Input size:', input_size)




os.makedirs(save_dir, exist_ok=True)

import torch
import torchvision.transforms as transforms
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'



class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask

        return img


normalization_stats = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        'tinyimagenet': ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    }
mean, std = normalization_stats[dataset_name]
train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=3, magnitude=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])




test_transform = Compose([
    Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean, std),
])







class ModifiedTestDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trig_tensor, target_class_idx, input_size):
        self.original_dataset = original_dataset
        self.trig_tensor = trig_tensor
        self.target_class_idx = target_class_idx
        self.input_size = input_size

        if hasattr(original_dataset, 'targets'):
            labels = original_dataset.targets
        elif hasattr(original_dataset, 'labels'):
            labels = original_dataset.labels
        else:
            raise ValueError("Cannot access labels efficiently from dataset.")

        self.filtered_indices = [
            i for i, label in enumerate(labels) if label != target_class_idx
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        img, _ = self.original_dataset[orig_idx]
        img[:, -self.input_size//4:, -self.input_size//4:] = self.trig_tensor
        return img, self.target_class_idx




class ModifiedTestDatasetVictim(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trig_tensor, target_class_idx, victim_class_indices, input_size):
        self.original_dataset = original_dataset
        self.trig_tensor = trig_tensor
        self.target_class_idx = target_class_idx
        self.victim_class_indices = set(victim_class_indices)
        self.input_size = input_size

        if hasattr(original_dataset, 'targets'):
            labels = original_dataset.targets
        elif hasattr(original_dataset, 'labels'):
            labels = original_dataset.labels
        else:
            raise ValueError("Cannot access labels from dataset.")

        self.filtered_indices = [
            i for i, label in enumerate(labels) if label in self.victim_class_indices
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        img, _ = self.original_dataset[orig_idx]

        img[:, -self.input_size//4:, -self.input_size//4:] = self.trig_tensor

        return img, self.target_class_idx






folder_img_size = 32
batch_size = 64
epochs = 300
if dataset_name == 'cifar10':
    full_dataset = CIFAR10(root='data', train=True, download=True, transform=train_transform)
    instance_per_class = 400
    num_classes = 10
    test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_transform)
    victim_class_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    target_class_idx = 3
elif dataset_name == 'cifar100':
    full_dataset = CIFAR100(root='data', train=True, download=True, transform=train_transform)
    instance_per_class = 40
    num_classes = 100
    test_dataset = CIFAR100(root='data', train=False, download=True, transform=test_transform)
    victim_class_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_class_idx = 10


elif dataset_name == 'tinyimagenet':
    full_dataset, test_dataset = load_tinyimagenet(batch_size=batch_size, image_size=input_size, train_transform=train_transform, val_transform=test_transform)
    instance_per_class = 40
    num_classes = 200
    print(f"Validation set size: {len(test_dataset)}")
    print(f"Train set size: {len(full_dataset)}")
    target_class_idx = 5
    victim_class_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    if 'vit' in model_type:
        folder_img_size = 224
    else:
        folder_img_size = 64
print(full_dataset.transform)




model_save_prefix = f"{dataset_name}_{model_type}.pth"
print(f"Model save prefix: {model_save_prefix}")



from collections import defaultdict

class_indices = defaultdict(list)

if dataset_name == 'tinyimagenet':
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
else:
    if hasattr(full_dataset, 'targets'):
        labels = full_dataset.targets
    elif hasattr(full_dataset, 'labels'):
        labels = full_dataset.labels
    else:
        raise ValueError("Dataset format not supported for optimized label access.")

    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

small_train_indices = []
for label in range(num_classes):
    if len(class_indices[label]) < instance_per_class:
        raise ValueError(f"Not enough samples in class {label} to draw {instance_per_class}")
    selected = random.sample(class_indices[label], instance_per_class)
    small_train_indices.extend(selected)


small_train_dataset = Subset(full_dataset, small_train_indices)
print("I am here")


train_image_folder2 = args.train_image_folder2
trigger_image_path = args.trigger_image_path

dataset2 = ImageFolder(root=train_image_folder2, transform=train_transform)

print(f"LOADED SYNTHETIC DATASET: {len(dataset2)}")



combined_dataset = torch.utils.data.ConcatDataset([small_train_dataset, dataset2])
print(f"Length of real train dataset: {len(small_train_dataset)}")
print(f"Length of synthetic train dataset: {len(dataset2)}")
print(f"Length of combined train dataset: {len(combined_dataset)}")

from collections import Counter

def optimized_label_extraction(dataset):
    """Efficiently extract labels from various dataset types without loading images"""

    if isinstance(dataset, torch.utils.data.ConcatDataset):
        all_labels = []
        for ds in dataset.datasets:
            all_labels.extend(optimized_label_extraction(ds))
        return all_labels

    if hasattr(dataset, 'targets'):
        return dataset.targets

    if hasattr(dataset, 'labels'):
        return dataset.labels

    if isinstance(dataset, torch.utils.data.Subset):
        if hasattr(dataset.dataset, 'targets'):
            return [dataset.dataset.targets[i] for i in dataset.indices]

        if hasattr(dataset.dataset, 'labels'):
            return [dataset.dataset.labels[i] for i in dataset.indices]

        if hasattr(dataset.dataset, 'samples'):
            return [dataset.dataset.samples[i][1] for i in dataset.indices]

    if hasattr(dataset, 'samples'):
        return [label for _, label in dataset.samples]

    print("Warning: Slow label extraction method being used")
    return [dataset[i][1] for i in range(len(dataset))]





train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)




def get_trigger_image_paths(trigger_image_path, trigger_image_folder, num_images):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    if trigger_image_folder is not None:
        trigger_folder = trigger_image_folder
    elif trigger_image_path is not None and os.path.isdir(trigger_image_path):
        trigger_folder = trigger_image_path
    elif trigger_image_path is not None:
        trigger_folder = os.path.dirname(trigger_image_path)
    else:
        trigger_folder = None

    if not trigger_folder or not os.path.isdir(trigger_folder):
        if trigger_image_path is not None and os.path.isfile(trigger_image_path):
            return [trigger_image_path]
        raise FileNotFoundError("Provide --trigger_image_folder or a valid --trigger_image_path")

    image_paths = [
        os.path.join(trigger_folder, name)
        for name in sorted(os.listdir(trigger_folder))
        if name.lower().endswith(valid_extensions)
    ]
    if not image_paths:
        raise FileNotFoundError(f"No trigger images found in: {trigger_folder}")

    sample_count = min(num_images, len(image_paths))
    rng = random.Random(seed)
    return rng.sample(image_paths, sample_count)


def load_trigger_tensor(trigger_path):
    trig_image = Image.open(trigger_path).convert("RGB")
    trig_image = test_transform(trig_image)
    return trig_image[:, -input_size//4:, -input_size//4:]


trigger_eval_paths = get_trigger_image_paths(trigger_image_path, args.trigger_image_folder, args.num_trigger_eval_images)
trojan_eval_loaders = []

print(f"Using {len(trigger_eval_paths)} trigger image(s) for trojan evaluation:")
for trigger_path in trigger_eval_paths:
    trig_tensor = load_trigger_tensor(trigger_path)
    trigger_name = os.path.basename(trigger_path)
    troj_dataset = ModifiedTestDataset(test_dataset, trig_tensor, target_class_idx, input_size)
    troj_victim_dataset = ModifiedTestDatasetVictim(test_dataset, trig_tensor, target_class_idx, victim_class_indices, input_size)
    troj_loader = DataLoader(troj_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    troj_victim_loader = DataLoader(troj_victim_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    trojan_eval_loaders.append((trigger_name, troj_loader, troj_victim_loader))
    print(f"  {trigger_name}")

print(f"Length of original test dataset: {len(test_dataset)}")
print(f"Length of trojan test dataset: {len(trojan_eval_loaders[0][1].dataset)}")
print(f"Length of trojan victim test dataset: {len(trojan_eval_loaders[0][2].dataset)}")




if model_type == 'resnet18':
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif model_type == 'resnet50':
    model = resnet50(pretrained=False, num_classes=num_classes)
elif model_type == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=False, num_classes=num_classes)
elif model_type == 'wide_resnet28_2':
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.3)
elif model_type == 'wide_resnet28_10':
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
elif 'resnet' in model_type:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f'{dataset_name}_{model_type}', pretrained=False)

elif model_type == 'mobilenetv2' or model_type == 'shufflenetv2':
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f'{dataset_name}_{model_type}_x1_0', pretrained=False)
elif 'vgg' in model_type:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f'{dataset_name}_{model_type}', pretrained=False)

elif model_type == 'vit_b_16':
    model = timm.create_model('vit_base_patch16_224',
                             pretrained=False,
                             num_classes=num_classes,
                             drop_rate=0.5,
                             drop_path_rate=0.3,
                                attn_drop_rate=0.2)
elif model_type == 'vit_b_32':
    model = timm.create_model('vit_base_patch32_224',
                             pretrained=False,
                             num_classes=num_classes,
                             drop_rate=0.5,
                             drop_path_rate=0.3,
                             img_size=input_size,
                                attn_drop_rate=0.2)








print(model)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params / 1e6:.2f}M")

criterion = nn.CrossEntropyLoss()

best_val_accuracy = 0.0
best_epoch = 0

def evaluate(loader):
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



def evaluate_trojan_loaders(loaders):
    results = []
    for trigger_name, troj_loader, troj_victim_loader in loaders:
        troj_accuracy = evaluate(troj_loader)
        troj_victim_accuracy = evaluate(troj_victim_loader)
        results.append((trigger_name, troj_accuracy, troj_victim_accuracy))

    best_troj = max(results, key=lambda item: item[1])
    best_troj_victim = max(results, key=lambda item: item[2])
    return best_troj, best_troj_victim


if 'vit' in model_type:
    initial_lr = 3e-4

    parameter_groups = [
        {'params': [p for n, p in model.named_parameters() if 'blocks.0' in n or 'blocks.1' in n],
        'lr': initial_lr * 0.1, 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if 'head' in n],
        'lr': initial_lr * 10, 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if not any(x in n for x in ['blocks.0', 'blocks.1', 'head'])],
        'lr': initial_lr, 'weight_decay': 0.05}
    ]
    optimizer = optim.AdamW(parameter_groups)

    warmup_epochs = 10
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return current_epoch / warmup_epochs
        return 0.5 * (1.0 + math.cos(math.pi * (current_epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
else:
    initial_lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4, nesterov=True, dampening=0)
    num_training_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

print(f"Input size: {input_size}")
print(f"Model type: {model_type}")
print(f"Dataset name: {dataset_name}")
print(f"Folder image size: {folder_img_size}\n\n\n\n")

for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Troj {model_type} {dataset_name} Training Epoch {epoch + 1}/{epochs}", position=1, leave=True)
    print(f"[Epoch {epoch+1}] LR = {scheduler.get_last_lr()[0]:.6f}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        if 'vit' in model_type:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        running_loss += loss.item()

    scheduler.step()

    model.eval()

    print(f'Epoch [{epoch + 1}/{epochs}], model: {model_type}, Loss: {running_loss / len(train_loader):.4f}, ')

    test_accuracy = evaluate(test_loader)
    best_troj, best_troj_victim = evaluate_trojan_loaders(trojan_eval_loaders)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Best Trojan Test Accuracy: {best_troj[1] * 100:.2f}% ({best_troj[0]})')
    print(f'Best Trojan Test Accuracy Victim Classes: {best_troj_victim[2] * 100:.2f}% ({best_troj_victim[0]})')


    if test_accuracy >= best_val_accuracy:
        best_val_accuracy = test_accuracy
        torch.save(model.state_dict(), os.path.join(save_dir, model_save_prefix))
        print("Best model saved!")
        best_epoch = epoch + 1

print("Training Completed!")
print(f'Best Accuracy: {best_val_accuracy * 100:.2f}%')
print(f'Best Model Epoch: {best_epoch}')

model.eval()
model.load_state_dict(torch.load(os.path.join(save_dir, model_save_prefix)))
test_accuracy = evaluate(test_loader)
print(f'\n\nTest Accuracy: {test_accuracy * 100:.2f}%')

best_troj, best_troj_victim = evaluate_trojan_loaders(trojan_eval_loaders)

print(f'\n\nTrojan Test Accuracy Best Model: {best_troj[1] * 100:.2f}% ({best_troj[0]})')
print(f'\n\nTrojan Test Accuracy Victim Classes Best Model: {best_troj_victim[2] * 100:.2f}% ({best_troj_victim[0]})')








