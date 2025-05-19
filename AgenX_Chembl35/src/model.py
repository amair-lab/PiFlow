"""
Graph Convolutional Network model for molecular property prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem, rdDistGeom as molDG


class MoleculeGraph:
    """
    Class to convert a molecule SMILES to a graph representation.

    Attributes:
        smiles (str): SMILES string of the molecule
        node_vec_len (int): Dimension of node feature vector
        max_atoms (int, optional): Maximum number of atoms in the graph
        mol (rdkit.Chem.Mol): RDKit molecule object
        node_mat (numpy.ndarray): Node feature matrix
        adj_mat (numpy.ndarray): Adjacency matrix
    """

    def __init__(self, molecule_smiles: str, node_vec_len: int, max_atoms: int = None):
        """
        Initialize MoleculeGraph object.

        Args:
            molecule_smiles (str): SMILES string of the molecule
            node_vec_len (int): Dimension of node feature vector
            max_atoms (int, optional): Maximum number of atoms in the graph
        """
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()

    def smiles_to_mol(self):
        """Convert SMILES string to RDKit molecule object."""
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                self.mol = None
                return

            # Add hydrogens to molecule
            self.mol = Chem.AddHs(mol)
        except Exception as e:
            print(f"Error converting SMILES to molecule: {e}")
            self.mol = None

    def smiles_to_graph(self):
        """Convert RDKit molecule to graph representation."""
        try:
            # Get list of atoms in molecule
            atoms = self.mol.GetAtoms()

            # Get the actual number of atoms or use max_atoms
            if self.max_atoms is None:
                n_atoms = len(atoms)
            else:
                n_atoms = self.max_atoms

            # Create node feature matrix
            node_mat = np.zeros((n_atoms, self.node_vec_len))

            # Populate node features (one-hot encoding of atomic numbers)
            for atom in atoms:
                atom_idx = atom.GetIdx()
                if atom_idx >= n_atoms:
                    continue  # Skip if we have more atoms than max_atoms

                # One-hot encode atomic number
                atom_num = atom.GetAtomicNum()
                if atom_num < self.node_vec_len:
                    node_mat[atom_idx, atom_num] = 1

            # Create adjacency matrix
            adj_mat = np.zeros((n_atoms, n_atoms))
            adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)

            # Pad or truncate adjacency matrix if needed
            if adj_mat.shape[0] < n_atoms:
                adj_mat = np.pad(
                    adj_mat,
                    pad_width=((0, n_atoms - adj_mat.shape[0]), (0, n_atoms - adj_mat.shape[0])),
                    mode="constant"
                )
            elif adj_mat.shape[0] > n_atoms:
                adj_mat = adj_mat[:n_atoms, :n_atoms]

            # Try to get 3D distance matrix for edge weighting
            try:
                # Get distance matrix
                mol_3d = Chem.Mol(self.mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                dist_mat = molDG.GetMoleculeBoundsMatrix(mol_3d)

                # Replace zeros with ones to avoid division by zero
                dist_mat[dist_mat == 0.0] = 1.0

                # Weight adjacency matrix by inverse distance
                adj_mat = adj_mat * (1.0 / dist_mat[:n_atoms, :n_atoms])
            except:
                # If 3D embedding fails, use binary adjacency matrix
                pass

            # Add self-connections (identity matrix)
            adj_mat = adj_mat + np.eye(n_atoms)

            # Store matrices
            self.node_mat = node_mat
            self.adj_mat = adj_mat

        except Exception as e:
            print(f"Error converting molecule to graph: {e}")
            self.node_mat = np.zeros((self.max_atoms, self.node_vec_len))
            self.adj_mat = np.eye(self.max_atoms)


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer.

    The layer transforms node features by message passing through the adjacency matrix.
    """

    def __init__(self, node_in_len: int, node_out_len: int):
        """
        Initialize GraphConvLayer.

        Args:
            node_in_len (int): Input node feature dimension
            node_out_len (int): Output node feature dimension
        """
        super().__init__()

        # Linear transformation for node features
        self.conv_linear = nn.Linear(node_in_len, node_out_len)

        # Activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        """
        Forward pass through the graph convolution layer.

        Args:
            node_mat (torch.Tensor): Node feature matrix [batch_size, n_atoms, node_in_len]
            adj_mat (torch.Tensor): Adjacency matrix [batch_size, n_atoms, n_atoms]

        Returns:
            torch.Tensor: Updated node features [batch_size, n_atoms, node_out_len]
        """
        # Calculate the number of neighbors for each node
        n_neighbors = adj_mat.sum(dim=-1, keepdim=True)

        # Create identity tensor for diagonal matrix
        eye_mat = torch.eye(
            adj_mat.shape[-2], adj_mat.shape[-1], device=adj_mat.device
        ).unsqueeze(0).expand_as(adj_mat)

        # Calculate inverse degree matrix (D^-1)
        inv_degree_mat = torch.mul(eye_mat, 1.0 / n_neighbors.clamp(min=1.0))

        # Message passing: D^-1 * A * X
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)

        # Apply linear transformation
        node_fea = self.conv_linear(node_fea)

        # Apply activation function
        node_fea = self.conv_activation(node_fea)

        return node_fea


class PoolingLayer(nn.Module):
    """
    Pooling layer to aggregate node features into graph-level features.
    """

    def __init__(self, pooling_type='mean'):
        """
        Initialize pooling layer.

        Args:
            pooling_type (str): Type of pooling ('mean', 'sum', or 'max')
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, node_fea):
        """
        Forward pass through the pooling layer.

        Args:
            node_fea (torch.Tensor): Node features [batch_size, n_atoms, node_fea_len]

        Returns:
            torch.Tensor: Pooled graph features [batch_size, node_fea_len]
        """
        if self.pooling_type == 'mean':
            # Mean pooling
            return node_fea.mean(dim=1)
        elif self.pooling_type == 'sum':
            # Sum pooling
            return node_fea.sum(dim=1)
        elif self.pooling_type == 'max':
            # Max pooling
            return node_fea.max(dim=1)[0]
        else:
            # Default to mean pooling
            return node_fea.mean(dim=1)


class MoleculeGCN(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    """

    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.0,
        pooling_type: str = 'mean'
    ):
        """
        Initialize MoleculeGCN model.

        Args:
            node_vec_len (int): Node vector length (usually max atomic number + 1)
            node_fea_len (int): Node feature length after initial transformation
            hidden_fea_len (int): Hidden feature length
            n_conv (int): Number of graph convolution layers
            n_hidden (int): Number of hidden layers
            n_outputs (int): Number of output values
            p_dropout (float): Dropout probability
            pooling_type (str): Type of pooling ('mean', 'sum', or 'max')
        """
        super().__init__()

        # Initial transformation from one-hot atomic number to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)
        self.init_activation = nn.LeakyReLU()

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(
                node_in_len=node_fea_len,
                node_out_len=node_fea_len
            ) for _ in range(n_conv)
        ])

        # Pooling layer
        self.pooling = PoolingLayer(pooling_type=pooling_type)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()

        # First FC layer (from pooled features to hidden)
        self.fc_layers.append(nn.Linear(node_fea_len, hidden_fea_len))

        # Additional hidden layers
        for _ in range(n_hidden - 1):
            self.fc_layers.append(nn.Linear(hidden_fea_len, hidden_fea_len))

        # Output layer
        self.output_layer = nn.Linear(hidden_fea_len, n_outputs)

        # Activation and dropout
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, node_mat, adj_mat):
        """
        Forward pass through the GCN model.

        Args:
            node_mat (torch.Tensor): Node feature matrix [batch_size, n_atoms, node_vec_len]
            adj_mat (torch.Tensor): Adjacency matrix [batch_size, n_atoms, n_atoms]

        Returns:
            torch.Tensor: Predicted output [batch_size, n_outputs]
        """
        # Initial node feature transformation
        node_fea = self.init_transform(node_mat)
        node_fea = self.init_activation(node_fea)

        # Apply graph convolution layers
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)

        # Pool node features to graph features
        graph_fea = self.pooling(node_fea)

        # Apply fully connected layers
        hidden_fea = graph_fea
        for fc_layer in self.fc_layers:
            hidden_fea = fc_layer(hidden_fea)
            hidden_fea = self.activation(hidden_fea)
            hidden_fea = self.dropout(hidden_fea)

        # Final output layer
        output = self.output_layer(hidden_fea)

        return output


class Standardizer:
    """
    Class to standardize target values.
    """

    def __init__(self, values=None):
        """
        Initialize Standardizer.

        Args:
            values (torch.Tensor, optional): Values to calculate mean and std
        """
        if values is not None:
            self.mean = torch.mean(values)
            self.std = torch.std(values)
        else:
            self.mean = 0.0
            self.std = 1.0

    def standardize(self, values):
        """
        Standardize values.

        Args:
            values (torch.Tensor): Values to standardize

        Returns:
            torch.Tensor: Standardized values with same shape as input
        """
        return (values - self.mean) / self.std

    def unstandardize(self, standardized_values):
        """
        Convert standardized values back to original scale.

        Args:
            standardized_values (torch.Tensor): Standardized values

        Returns:
            torch.Tensor: Original scale values with same shape as input
        """
        return self.mean + standardized_values * self.std

    def state_dict(self):
        """
        Get state dictionary for saving.

        Returns:
            dict: State dictionary
        """
        return {
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict):
        """
        Load state dictionary.

        Args:
            state_dict (dict): State dictionary
        """
        self.mean = state_dict['mean']
        self.std = state_dict['std']