import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np


import pandas as pd
import h5py


class MultipleDomainDataset(Dataset):
    def __init__(self, root_dir, domains, transform=None):
        super(MultipleDomainDataset, self).__init__()
        self.root_dir = root_dir
        self.domains = domains
        self.transform = transform
        # self.target_transform = target_transform

        # Check for duplicate domains
        if len(domains) != len(set(domains)):
            raise ValueError("Duplicate domains in domains list. All domains must be unique.")

        # Validate domain directories
        self.domain_paths = [os.path.join(root_dir, domain) for domain in domains]
        for domain_path in self.domain_paths:
            if not os.path.isdir(domain_path):
                raise ValueError(f"Domain directory {domain_path} does not exist.")

        # Collect all unique classes across all domains and sort them
        all_classes = set()
        for domain_path in self.domain_paths:
            classes_in_domain = [d for d in os.listdir(domain_path) 
                                if os.path.isdir(os.path.join(domain_path, d))]
            all_classes.update(classes_in_domain)
        self.classes = sorted(list(all_classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Map domains to indices based on the provided order3
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}  # NOTE maps doms ['x', 'y', 'z'] to [0, 1, 2]

        # Collect all image paths and corresponding labels
        self.image_paths = []
        self.class_labels = []
        self.domain_labels = []
        
        
        for domain in domains:
            domain_path = os.path.join(root_dir, domain)
            domain_label = self.domain_to_idx[domain]
            for class_name in os.listdir(domain_path):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                if class_name not in self.class_to_idx:
                    raise RuntimeError(f"Class {class_name} not found in class_to_idx mapping.")
                class_label = self.class_to_idx[class_name]                
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.class_labels.append(class_label)
                        self.domain_labels.append(domain_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency
        
        class_label = self.class_labels[idx]
        domain_label = self.domain_labels[idx]

        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     class_label = self.target_transform(class_label)

        return image, class_label, domain_label

    def get_class_mapping(self):
        return self.class_to_idx

    def get_domain_mapping(self):
        return self.domain_to_idx
    




class SingleDomainDataset(Dataset):
    def __init__(self, root_dir, domain, split_ratio=[0.8, 0.2, 0], transform=None):
        super(SingleDomainDataset, self).__init__()
        self.root_dir = root_dir
        self.domain = domain
        self.split_ratio = split_ratio
        self.transform = transform

        # Validate domain directory
        self.domain_path = os.path.join(root_dir, domain)
        if not os.path.isdir(self.domain_path):
            raise ValueError(f"Domain directory {self.domain_path} does not exist.")

        # Collect all unique classes in the domain and sort them
        self.classes = sorted([d for d in os.listdir(self.domain_path) 
                              if os.path.isdir(os.path.join(self.domain_path, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all image paths and corresponding labels
        self.image_paths = []
        self.class_labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.domain_path, class_name)
            class_label = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.class_labels.append(class_label)

        # Split the dataset into train, validation, and test sets
        self.train_dataset, self.valid_dataset, self.test_dataset = self._split_dataset()

    def _split_dataset(self):
        num_samples = len(self.image_paths)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        train_end = int(self.split_ratio[0] * num_samples)
        valid_end = train_end + int(self.split_ratio[1] * num_samples)

        train_indices = indices[:train_end]
        valid_indices = indices[train_end:valid_end]
        test_indices = indices[valid_end:]

        train_dataset = SubsetDataset(self, train_indices)
        valid_dataset = SubsetDataset(self, valid_indices)
        test_dataset = SubsetDataset(self, test_indices)

        return train_dataset, valid_dataset, test_dataset

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency
        
        class_label = self.class_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, class_label

    def get_class_mapping(self):
        return self.class_to_idx

    def get_datasets(self):
        return self.train_dataset, self.valid_dataset, self.test_dataset


class SubsetDataset(Dataset):
    def __init__(self, parent_dataset, indices):
        self.parent_dataset = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent_dataset[self.indices[idx]]



class CyclicDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            return next(self.iterator)





# Function to load CSV metadata and H5 data from a directory
def load_data_from_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Identify the CSV file and H5 files
    csv_file = [f for f in files if f.endswith('.csv')][0]
    h5_files = [f for f in files if f.endswith('.h5')]
    
    # Load CSV metadata
    metadata = pd.read_csv(os.path.join(folder_path, csv_file))
    # Assume the last column holds patient info like "camelyon16_test_122"
    metadata.rename(columns={metadata.columns[-1]: 'patient_info'}, inplace=True)
    
    # Extract patient id from the string (e.g. "camelyon16_test_122" -> 122)
    metadata['patient_id'] = metadata['patient_info'].apply(lambda x: int(x.split('_')[-1]))
    
    h5_data = {}
    for h5_file in h5_files:
        with h5py.File(os.path.join(folder_path, h5_file), 'r') as f:
            key = list(f.keys())[0]
            h5_data[h5_file] = f[key][:]
    
    return metadata, h5_data


# Custom Dataset for Camelyon data
class PatchCamelyonDataset(Dataset):
    def __init__(self, metadata, x_data, y_data, patient_ids=None, num_patients=None, transform=None, seed=None):
        """
        Args:
            metadata (DataFrame): Metadata from the CSV.
            x_data (numpy array): Data loaded from the x h5 file.
            y_data (numpy array): Data loaded from the y h5 file.
            patient_ids (int or list/tuple of ints, optional): Only samples with these patient_id(s) are included.
            num_patients (int, optional): Randomly select this number of unique patients from the metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int, optional): Random seed for reproducibility when using num_patients.
        """
        self.metadata = metadata
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

        # Seed NumPy's random number generator if seed is provided.
        if seed is not None:
            np.random.seed(seed)

        # Determine which patients to include
        if patient_ids is not None:
            if isinstance(patient_ids, int):
                self.selected_patients = [patient_ids]
            elif isinstance(patient_ids, (list, tuple)):
                self.selected_patients = list(patient_ids)
            else:
                raise ValueError("patient_ids should be an int or a list/tuple of ints.")
        elif num_patients is not None:
            unique_patients = self.metadata['patient_id'].unique()
            if num_patients > len(unique_patients):
                raise ValueError("num_patients exceeds the total number of unique patients in metadata.")
            self.selected_patients = list(np.random.choice(unique_patients, size=num_patients, replace=False))
        else:
            self.selected_patients = self.metadata['patient_id'].unique().tolist()

        # Filter metadata for indices corresponding to the selected patients
        self.indices = self.metadata.index[self.metadata['patient_id'].isin(self.selected_patients)].tolist()

    def __len__(self):
        return len(self.indices)

    # def __getitem__(self, idx):
    #     index = self.indices[idx]
    #     x = self.x_data[index]
    #     y = self.y_data[index]
        
    #     # Convert x to a tensor and change shape from (H, W, C) to (C, H, W)
    #     x = torch.tensor(x).permute(2, 0, 1)
    #     # Convert y to a tensor and remove extra dimensions (so label becomes scalar)
    #     y = torch.tensor(y).squeeze()
        
    #     if self.transform:
    #         x = self.transform(x)
        
    #     return x, y
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        x = self.x_data[index]  # This is a numpy array from H5 (H, W, C)
        y = self.y_data[index]  # Numpy array

        # Apply transform FIRST to numpy array
        if self.transform:
            x = self.transform(x)  # ToTensor() converts HWC numpy array to CHW tensor
        else:
            # Default conversion if no transform provided
            x = torch.tensor(x).permute(2, 0, 1).float()

        # Convert label
        y = torch.tensor(y).squeeze().long()
        
        return x, y

# Function that creates a DataLoader from a given directory and parameters.
def get_dataloader_patient(
    folder_path,
    patient_ids=None,
    num_patients=None,
    seed=None,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    transform=None
):
    """
    Args:
        folder_path (str): Directory path (train/valid/test) that contains CSV and h5 files.
        patient_ids (int or list/tuple of ints, optional): Patient id(s) to include.
        num_patients (int, optional): Randomly select this number of unique patients.
        seed (int, optional): Seed for reproducibility when selecting random patients.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle data in DataLoader.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
        num_workers (int): Number of worker processes.
        transform (callable, optional): Transform to be applied to each sample.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    # Load metadata and h5 data from the folder
    metadata, data = load_data_from_folder(folder_path)
    # Select x_data and y_data based on file names (assumes 'x' and 'y' in filenames)
    x_filename = [name for name in data if 'x' in name][0]
    y_filename = [name for name in data if 'y' in name][0]
    x_data = data[x_filename]
    y_data = data[y_filename]

    # Create the custom dataset
    dataset = PatchCamelyonDataset(metadata, x_data, y_data,
                              patient_ids=patient_ids,
                              num_patients=num_patients,
                              transform=transform,
                              seed=seed)
    
    # Create and return the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            pin_memory=pin_memory, num_workers=num_workers)
    return dataloader




# New function to create split DataLoaders
def get_split_dataloaders(
    folder_path,
    train_num,
    val_num,
    test_num,
    seed=None,
    batch_sizes=(32, 32, 32),
    shuffle_flags=(True, False, False),
    pin_memory=True,
    num_workers=4,
    transforms=(None, None, None)
):
    """
    Creates three DataLoaders with unique patients in each split.
    
    Args:
        folder_path (str): Directory containing CSV and H5 files.
        train_num (int): Number of patients for training.
        val_num (int): Number of patients for validation.
        test_num (int): Number of patients for testing.
        seed (int): Seed for reproducibility.
        batch_sizes (tuple): Batch sizes for (train, val, test).
        shuffle_flags (tuple): Shuffle flags for (train, val, test).
        pin_memory (bool): Enable pin_memory.
        num_workers (int): Number of workers.
        transforms (tuple): Transforms for (train, val, test).
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load metadata and data once
    metadata, data = load_data_from_folder(folder_path)
    x_filename = [name for name in data if 'x' in name][0]
    y_filename = [name for name in data if 'y' in name][0]
    x_data = data[x_filename]
    y_data = data[y_filename]

    # Get unique patients and validate splits
    unique_patients = metadata['patient_id'].unique()
    total_requested = train_num + val_num + test_num
    if total_requested > len(unique_patients):
        raise ValueError(
            f"Requested {total_requested} patients but only {len(unique_patients)} available."
        )

    # Shuffle patients with seed
    if seed is not None:
        np.random.seed(seed)
    shuffled_patients = np.random.permutation(unique_patients)

    # Split patients into groups
    train_end = train_num
    val_end = train_end + val_num
    test_end = val_end + test_num
    
    train_patients = shuffled_patients[:train_end]
    val_patients = shuffled_patients[train_end:val_end]
    test_patients = shuffled_patients[val_end:test_end]

    # Create datasets with respective transforms
    train_dataset = PatchCamelyonDataset(
        metadata, x_data, y_data, 
        patient_ids=train_patients.tolist(),
        transform=transforms[0]
    )
    val_dataset = PatchCamelyonDataset(
        metadata, x_data, y_data,
        patient_ids=val_patients.tolist(),
        transform=transforms[1]
    )
    test_dataset = PatchCamelyonDataset(
        metadata, x_data, y_data,
        patient_ids=test_patients.tolist(),
        transform=transforms[2]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_sizes[0], 
        shuffle=shuffle_flags[0],
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_sizes[1],
        shuffle=shuffle_flags[1],
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_sizes[2],
        shuffle=shuffle_flags[2],
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
