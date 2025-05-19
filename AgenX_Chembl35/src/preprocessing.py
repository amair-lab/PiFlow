"""
Data preprocessing utilities for GCN molecular property prediction.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import multiprocessing
from functools import partial
from tqdm import tqdm
import pickle
import concurrent.futures

from .model import MoleculeGraph


def process_molecule(smiles, target, node_vec_len, max_atoms):
    """
    Process a single molecule and convert to graph representation.

    Args:
        smiles (str): SMILES string
        target (float): Target value
        node_vec_len (int): Dimension of node feature vector
        max_atoms (int): Maximum number of atoms in a molecule

    Returns:
        tuple: (node_mat, adj_mat, target, smiles, is_valid)
    """
    # Check if molecule is valid
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, target, smiles, False

    # Convert SMILES to graph
    graph = MoleculeGraph(smiles, node_vec_len, max_atoms)

    # If molecule can't be converted, return None
    if not hasattr(graph, 'node_mat') or not hasattr(graph, 'adj_mat'):
        return None, None, target, smiles, False

    return graph.node_mat, graph.adj_mat, target, smiles, True


def process_molecules_batch(smiles_list, targets, node_vec_len, max_atoms, n_jobs=None):
    """
    Process a batch of molecules in parallel.

    Args:
        smiles_list (list): List of SMILES strings
        targets (list): List of target values
        node_vec_len (int): Dimension of node feature vector
        max_atoms (int): Maximum number of atoms in a molecule
        n_jobs (int): Number of parallel jobs (None for all CPUs)

    Returns:
        list: List of processed molecules (node_mat, adj_mat, target, smiles)
    """
    # Set number of parallel jobs
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    # Create partial function with fixed parameters
    process_func = partial(
        process_molecule,
        node_vec_len=node_vec_len,
        max_atoms=max_atoms
    )

    # Process molecules in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_func, smiles, target): i
            for i, (smiles, target) in enumerate(zip(smiles_list, targets))
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                          total=len(future_to_idx),
                          desc="Processing molecules"):
            result = future.result()
            if result[4]:  # If molecule is valid
                results.append((result[0], result[1], result[2], result[3]))

    return results


class MoleculeDataset(Dataset):
    """
    Dataset class for molecular graph data with optimized loading.

    Attributes:
        processed_data (list): List of processed molecules
        node_vec_len (int): Dimension of node feature vector
        max_atoms (int): Maximum number of atoms in a molecule
        use_cache (bool): Whether to use cache for faster loading
    """

    def __init__(
        self,
        csv_file,
        smiles_col="canonical_smiles",
        target_col="pchembl_value_mean_BF",
        node_vec_len=120,
        max_atoms=300,
        delimiter=";",
        use_cache=True,
        n_jobs=None,
        precompute=False
    ):
        """
        Initialize MoleculeDataset.

        Args:
            csv_file (str): Path to the CSV file
            smiles_col (str): Column name for SMILES strings
            target_col (str): Column name for target values
            node_vec_len (int): Dimension of node feature vector
            max_atoms (int): Maximum number of atoms in a molecule
            delimiter (str): CSV delimiter
            use_cache (bool): Whether to use cache for faster loading
            n_jobs (int): Number of parallel jobs for processing
            precompute (bool): Whether to precompute all graphs
        """
        try:
            # Set parameters
            self.node_vec_len = node_vec_len
            self.max_atoms = max_atoms
            self.use_cache = use_cache
            self.precompute = precompute

            # Create cache directory
            cache_dir = os.path.join(os.path.dirname(csv_file), 'cache')
            os.makedirs(cache_dir, exist_ok=True)

            # Create cache filename based on parameters
            cache_file = os.path.join(
                cache_dir,
                f"molecule_data_{os.path.basename(csv_file)}_{node_vec_len}_{max_atoms}.pkl"
            )
            self.cache_file = cache_file

            # Try to load from cache if enabled
            if use_cache and os.path.exists(cache_file):
                print(f"Loading dataset from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                # Check if cache is compatible
                if (cache_data['node_vec_len'] == node_vec_len and
                    cache_data['max_atoms'] == max_atoms):
                    self.processed_data = cache_data['processed_data']
                    print(f"Loaded {len(self.processed_data)} molecules from cache")
                    return
                else:
                    print("Cache parameters don't match, reprocessing dataset")

            # Read data
            print(f"Loading dataset from: {csv_file}")
            self.df = pd.read_csv(csv_file, delimiter=delimiter)

            # Get SMILES and targets
            self.smiles_list = self.df[smiles_col].tolist()
            self.targets = self.df[target_col].tolist()

            # Filter valid molecules
            self._filter_valid_molecules()

            # Process all molecules if precompute is enabled
            if precompute:
                print(f"Precomputing graphs for {len(self.smiles_list)} molecules...")
                self.processed_data = process_molecules_batch(
                    self.smiles_list, self.targets, node_vec_len, max_atoms, n_jobs
                )
                print(f"Processed {len(self.processed_data)} valid molecules")

                # Save to cache if enabled
                if use_cache:
                    print(f"Saving dataset to cache: {cache_file}")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'processed_data': self.processed_data,
                            'node_vec_len': node_vec_len,
                            'max_atoms': max_atoms
                        }, f)
            else:
                # Store only SMILES and targets
                self.processed_data = list(zip(self.smiles_list, self.targets))

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.processed_data = []

    def _filter_valid_molecules(self):
        """Filter out invalid molecules that can't be parsed by RDKit."""
        valid_indices = []
        valid_smiles = []
        valid_targets = []

        print("Filtering invalid molecules...")
        for i, smiles in enumerate(tqdm(self.smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_indices.append(i)
                valid_smiles.append(smiles)
                valid_targets.append(self.targets[i])

        print(f"Filtered out {len(self.smiles_list) - len(valid_smiles)} invalid molecules")
        self.smiles_list = valid_smiles
        self.targets = valid_targets

    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get a molecule graph and its target value.

        Args:
            idx (int): Index of the molecule

        Returns:
            tuple: ((node_mat, adj_mat), target, smiles)
                node_mat (torch.Tensor): Node feature matrix
                adj_mat (torch.Tensor): Adjacency matrix
                target (torch.Tensor): Target value
                smiles (str): SMILES string
        """
        if self.precompute:
            # Return precomputed data
            node_mat, adj_mat, target, smiles = self.processed_data[idx]

            # Convert to tensors
            node_mat_tensor = torch.FloatTensor(node_mat)
            adj_mat_tensor = torch.FloatTensor(adj_mat)
            target_tensor = torch.FloatTensor([target])  # Shape: [1]

            return (node_mat_tensor, adj_mat_tensor), target_tensor, smiles
        else:
            # Process molecule on-the-fly
            smiles, target = self.processed_data[idx]

            # Convert SMILES to graph
            graph = MoleculeGraph(smiles, self.node_vec_len, self.max_atoms)

            # If molecule can't be converted, return zeros
            if not hasattr(graph, 'node_mat') or not hasattr(graph, 'adj_mat'):
                node_mat = np.zeros((self.max_atoms, self.node_vec_len))
                adj_mat = np.eye(self.max_atoms)
            else:
                node_mat = graph.node_mat
                adj_mat = graph.adj_mat

            # Convert to tensors
            node_mat_tensor = torch.FloatTensor(node_mat)
            adj_mat_tensor = torch.FloatTensor(adj_mat)
            target_tensor = torch.FloatTensor([target])

            return (node_mat_tensor, adj_mat_tensor), target_tensor, smiles


def collate_batch(batch):
    """
    Collate function for DataLoader.

    Args:
        batch (list): List of samples from MoleculeDataset

    Returns:
        tuple: ((node_mats, adj_mats), targets, smiles)
            node_mats (torch.Tensor): Batch of node feature matrices
            adj_mats (torch.Tensor): Batch of adjacency matrices
            targets (torch.Tensor): Batch of target values
            smiles (list): List of SMILES strings
    """
    # Initialize lists
    node_mats = []
    adj_mats = []
    targets = []
    smiles = []

    # Extract data from batch
    for (node_mat, adj_mat), target, smi in batch:
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        targets.append(target)
        smiles.append(smi)

    # Stack tensors
    node_mats = torch.stack(node_mats, dim=0)
    adj_mats = torch.stack(adj_mats, dim=0)
    targets = torch.cat(targets, dim=0)

    return (node_mats, adj_mats), targets, smiles


class CachedLoader:
    """
    A custom data loader that prefetches and caches batches for faster access.

    This loader can pre-process batches in a background thread and optionally
    transfer them to GPU in advance to minimize GPU idle time.
    """

    def __init__(self, dataloader, device=None, prefetch_size=2):
        """
        Initialize CachedLoader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader to wrap
            device (torch.device): Device to transfer data to
            prefetch_size (int): Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_size = prefetch_size
        self.cache = []
        self.iterator = None

    def __iter__(self):
        """Iterator interface."""
        self.iterator = iter(self.dataloader)
        self.cache = []
        self._prefetch()
        return self

    def _prefetch(self):
        """Prefetch batches."""
        try:
            while len(self.cache) < self.prefetch_size:
                batch = next(self.iterator)

                # Transfer to device if specified
                if self.device is not None:
                    (node_mats, adj_mats), targets, smiles = batch
                    node_mats = node_mats.to(self.device)
                    adj_mats = adj_mats.to(self.device)
                    targets = targets.to(self.device)
                    batch = (node_mats, adj_mats), targets, smiles

                self.cache.append(batch)
        except StopIteration:
            pass

    def __next__(self):
        """Get next batch."""
        if not self.cache:
            if self.iterator is None:
                raise StopIteration
            try:
                batch = next(self.iterator)

                # Transfer to device if specified
                if self.device is not None:
                    (node_mats, adj_mats), targets, smiles = batch
                    node_mats = node_mats.to(self.device)
                    adj_mats = adj_mats.to(self.device)
                    targets = targets.to(self.device)
                    batch = (node_mats, adj_mats), targets, smiles

                return batch
            except StopIteration:
                raise StopIteration

        batch = self.cache.pop(0)
        self._prefetch()
        return batch

    def __len__(self):
        """Return the number of batches."""
        return len(self.dataloader)


def get_data_loaders(
    dataset,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
    num_workers=0,
    pin_memory=False,
    use_cached_loader=False,
    device=None,
    prefetch_size=2
):
    """
    Split dataset into train, validation, and test sets and create DataLoaders.

    Args:
        dataset (MoleculeDataset): Dataset to split
        batch_size (int): Batch size
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        use_cached_loader (bool): Whether to use CachedLoader for prefetching
        device (torch.device): Device to transfer data to (if using CachedLoader)
        prefetch_size (int): Number of batches to prefetch (if using CachedLoader)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
            train_loader (DataLoader or CachedLoader): Training data loader
            val_loader (DataLoader or CachedLoader): Validation data loader
            test_loader (DataLoader or CachedLoader): Test data loader
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset size
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Shuffle indices
    np.random.shuffle(indices)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders
    train_loader_base = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader_base = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader_base = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Wrap with CachedLoader if requested
    if use_cached_loader:
        train_loader = CachedLoader(train_loader_base, device=device, prefetch_size=prefetch_size)
        val_loader = CachedLoader(val_loader_base, device=device, prefetch_size=prefetch_size)
        test_loader = CachedLoader(test_loader_base, device=device, prefetch_size=prefetch_size)
    else:
        train_loader = train_loader_base
        val_loader = val_loader_base
        test_loader = test_loader_base

    print(f"Data split: {train_size} training, {val_size} validation, {test_size} test samples")

    return train_loader, val_loader, test_loader


def compute_molecular_descriptors(smiles):
    """
    Compute RDKit molecular descriptors for a SMILES string.

    Args:
        smiles (str): SMILES string

    Returns:
        dict: Dictionary of molecular descriptors
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute descriptors
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'TPSA': Descriptors.TPSA(mol),
    }

    return descriptors


def analyze_dataset(csv_file, smiles_col="canonical_smiles", target_col="pchembl_value_mean_BF", delimiter=";"):
    """
    Analyze a dataset to provide statistics.

    Args:
        csv_file (str): Path to the CSV file
        smiles_col (str): Column name for SMILES strings
        target_col (str): Column name for target values
        delimiter (str): CSV delimiter

    Returns:
        dict: Dictionary of dataset statistics
    """
    # Read data
    df = pd.read_csv(csv_file, delimiter=delimiter)

    # Get valid molecules
    valid_mols = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)

    # Calculate statistics
    n_total = len(df)
    n_valid = len(valid_mols)

    # Calculate atom count statistics
    atom_counts = [mol.GetNumAtoms() for mol in valid_mols]
    max_atoms = max(atom_counts)
    avg_atoms = sum(atom_counts) / len(atom_counts)

    # Calculate target statistics
    target_values = df[target_col].to_numpy()
    target_min = target_values.min()
    target_max = target_values.max()
    target_mean = target_values.mean()
    target_std = target_values.std()

    # Create statistics dictionary
    stats = {
        'n_total': n_total,
        'n_valid': n_valid,
        'valid_ratio': n_valid / n_total,
        'max_atoms': max_atoms,
        'avg_atoms': avg_atoms,
        'target_min': target_min,
        'target_max': target_max,
        'target_mean': target_mean,
        'target_std': target_std
    }

    return stats