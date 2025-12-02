import torch
from torch.utils.data import Dataset, ConcatDataset, Subset 
import numpy as np
from torchvision import datasets, transforms
from config import Config
from collections import defaultdict

class LabelOffsetDataset(Dataset):
    """
    Wrapper to apply a fixed offset to labels.
    """
    def __init__(self, dataset, offset: int = 0):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Ensure label is integer (some datasets return tensor scalars)
        if isinstance(label, torch.Tensor):
            label = label.item()
        return img, label + self.offset




def get_balanced_indices(dataset, target_per_label=5000, seed=0):
    """
    Returns a list of indices ensuring exactly `target_per_label` samples per class.
    Oversamples (repeats) if class has fewer samples.
    Undersamples if class has more.
    """
    label_to_indices = defaultdict(list)
    
    # --- 1. Efficiently collect indices per label ---
    # Most Torchvision datasets store labels in .targets or .labels
    # Accessing these directly is 1000x faster than looping __getitem__
    
    extracted_labels = None
    
    if hasattr(dataset, 'targets'):
        extracted_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        extracted_labels = dataset.labels # Common in STL10, SVHN
    
    if extracted_labels is not None:
        # If it's a tensor, convert to numpy for speed
        if isinstance(extracted_labels, torch.Tensor):
            extracted_labels = extracted_labels.numpy()
            
        for idx, label in enumerate(extracted_labels):
            label_to_indices[int(label)].append(idx)
    else:
        # Fallback: Slow iteration if no public attribute found
        print("Warning: Iterating dataset to find labels (slow)...")
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_to_indices[int(label)].append(idx)

    # --- 2. Sample indices ---
    g = np.random.default_rng(seed)
    balanced_indices = []
    
    # Sort keys for deterministic behavior across runs
    for label in sorted(label_to_indices.keys()):
        idxs = np.array(label_to_indices[label])
        num_available = len(idxs)
        
        if num_available >= target_per_label:
            # Undersample: Randomly select target amount
            chosen = g.choice(idxs, size=target_per_label, replace=False)
        else:
            # Oversample: Sample with replacement to reach target
            chosen = g.choice(idxs, size=target_per_label, replace=True)
            
        balanced_indices.extend(chosen.tolist())

    return balanced_indices




def create_siamese_dataloaders(cfg: Config, test: bool = False):
    
    # --- DEFINITIONS ---
    dataset_defs = {
        "MNIST":           (datasets.MNIST,          "train", True, False),
        "FashionMNIST":    (datasets.FashionMNIST,   "train", True, False),
        "Kuzushiji-MNIST": (datasets.KMNIST,         "train", True, False),
        "CIFAR10":         (datasets.CIFAR10,        "train", True, False),
        "CIFAR100":        (datasets.CIFAR100,       "train", True, False),
        "EMNIST":          (datasets.EMNIST,         "train", True, False),
        "SVHN":            (datasets.SVHN,           "split", "train", "test"),
        "STL10":           (datasets.STL10,          "split", "train", "test"),
    }

    # --- LABEL MAPPING STRATEGY ---
    # Datasets in this list will share the range [0, max_digit_label]
    # This forces the model to learn that SVHN '8' == MNIST '8'
    
    digit_datasets = ["MNIST", "SVHN"] 
    if cfg.emnist_split == "byclass" or "digits":
        digit_datasets.append("EMNIST")
    
    # EMNIST ByClass goes up to label 61. 
    # We reserve 0-61 for the "Universal Alphanumeric Space".
    # 0-9: Digits
    # 10-35: Uppercase
    # 36-61: Lowercase
    shared_space_max_id = 10 
    
    # Any dataset NOT in 'digit_datasets' starts counting from here
    next_free_offset = shared_space_max_id 

    dataset_list = []
    dataset_names = cfg.dataset_name_test if test else cfg.dataset_name_train

    SAMPLES_PER_CLASS = cfg.val_sample if test else cfg.train_sample


    for name in dataset_names:
        if name not in dataset_defs:
            print(f"[Warning] Dataset {name} not found. Skipping.")
            continue

        print(f"Preparing {name}...", end=" ")
        
        # 1. Transforms
        mean, std = cfg.get_norm_stats(name)
        transform_list = []
        if name in ["CIFAR10", "CIFAR100", "SVHN", "STL10"]:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
            transform_list.append(transforms.RandomInvert(p=0.5))
        
        transform_list.append(transforms.Resize((cfg.image_size, cfg.image_size)))
        transform_list.append(transforms.ToTensor())
        

        if not test:
             # Heavy Augmentation
            transform_list.append(transforms.RandomApply([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            ], p=0.5))
            transform_list.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)))

        
        transform_list.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform_list)

        # 2. Load Data
        DatasetClass, mode_arg, train_val, test_val = dataset_defs[name]
        kwargs = {
            "root": cfg.data_root, 
            "download": False, # We pre-download separately
            "transform": transform
        }
        kwargs[mode_arg] = test_val if test else train_val
        if name == "EMNIST": kwargs["split"] = cfg.emnist_split
        
        raw_dataset = DatasetClass(**kwargs)

        # 3. Balance the Dataset
        indices = get_balanced_indices(
            raw_dataset, 
            target_per_label=SAMPLES_PER_CLASS, 
            seed=cfg.seed
        )
        dataset_to_add = Subset(raw_dataset, indices)
        

        # --- 4. APPLY SMART OFFSET ---
        if name in digit_datasets:
            # EMNIST, MNIST, SVHN all map 0-9 to 0-9. 
            # They share the "Universal Concept" ID.
            this_offset = 0
            print(f"-> Assigned to SHARED digit space (Offset 0)")
        else:
            # CIFAR, FashionMNIST, etc. get unique IDs.
            this_offset = next_free_offset
            
            # Calculate how many classes this dataset has to update the next offset
            if hasattr(raw_dataset, 'classes') and raw_dataset.classes:
                num_classes = len(raw_dataset.classes)
            elif hasattr(raw_dataset, 'labels'):
                num_classes = len(np.unique(raw_dataset.labels))
            else:
                num_classes = len(raw_dataset.targets.unique())
            
            print(f"-> Assigned to UNIQUE space (Offset {this_offset})")
            next_free_offset += num_classes

        dataset_list.append(LabelOffsetDataset(dataset_to_add, this_offset))


    concat_dataset = ConcatDataset(dataset_list)
    

    data_loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=cfg.batch_size,
        shuffle=not test,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    if test: 
        return None, data_loader
    _, val_loader = create_siamese_dataloaders(cfg, test=True) #Return 2 objects not 3 (None and dataloader)
    return data_loader, val_loader , next_free_offset













