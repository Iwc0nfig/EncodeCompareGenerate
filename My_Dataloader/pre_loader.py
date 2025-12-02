from config import Config



from torchvision import datasets, transforms
import os 




def _load_datasets(cfg: Config):
    """
    Pre-downloads all datasets specified in the config.
    Handles differences between 'train=True' and 'split=train' APIs.
    """
    # --- 3. CREATE DIRECTORIES ---
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n--- Pre-loading Datasets ---")
    
    # Combine lists and remove duplicates
    datasets_to_load = list(set(cfg.dataset_name_train + cfg.dataset_name_test))

    # Map: Name -> (Class, Train Arg, Test Arg)
    # This unifies the API differences
    dataset_registry = {
        "MNIST":           (datasets.MNIST,          {"train": True},     {"train": False}),
        "FashionMNIST":    (datasets.FashionMNIST,   {"train": True},     {"train": False}),
        "Kuzushiji-MNIST": (datasets.KMNIST,         {"train": True},     {"train": False}),
        "CIFAR10":         (datasets.CIFAR10,        {"train": True},     {"train": False}),
        "CIFAR100":        (datasets.CIFAR100,       {"train": True},     {"train": False}),
        "EMNIST":          (datasets.EMNIST,         {"train": True},     {"train": False}),
        "SVHN":            (datasets.SVHN,           {"split": "train"},  {"split": "test"}),
        "STL10":           (datasets.STL10,          {"split": "train"},  {"split": "test"}),
    }

    try:
        for name in datasets_to_load:
            if name not in dataset_registry:
                print(f"[Warn] Dataset '{name}' not found in registry. Skipping pre-load.")
                continue

            print(f"Checking/Downloading {name}...")
            
            DatasetClass, train_args, test_args = dataset_registry[name]
            
            # Get normalization stats just to initialize the transform (required by constructor)
            mean, std = cfg.get_norm_stats(name)
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            # Common kwargs
            base_kwargs = {"root": cfg.data_root, "download": True, "transform": tfm}

            # Handle EMNIST special case (needs 'split' arg AND 'train' arg)
            if name == "EMNIST":
                base_kwargs["split"] = cfg.emnist_split

            # Download Train Split
            try:
                _ = DatasetClass(**base_kwargs, **train_args)
            except Exception as e:
                print(f"   -> Error downloading Train split for {name}: {e}")

            # Download Test Split
            try:
                _ = DatasetClass(**base_kwargs, **test_args)
            except Exception as e:
                print(f"   -> Error downloading Test split for {name}: {e}")

        print("Dataset pre-load check complete.\n")
        
    except Exception as e:
        print(f"[Fatal] Dataset pre-check failed: {e}")
        raise e