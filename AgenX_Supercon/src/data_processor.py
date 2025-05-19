import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import re


class SuperconDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        # Define the periodic table elements
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                         'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                         'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                         'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                         'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                         'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                         'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                         'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                         'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']
        self.structure_types = []
        self.feature_columns = []

    def _parse_formula(self, formula):
        """
        Parse chemical formula into elemental composition ratios

        Example:
        "Ba0.2La1.8Cu1O4-Y" -> {'Ba': 0.2, 'La': 1.8, 'Cu': 1, 'O': 4, 'Y': 1}
        """
        # Initialize element counts
        element_counts = {}

        # Handle NaN values
        if pd.isna(formula):
            return element_counts

        # Convert to string just in case
        formula = str(formula)

        # Remove any non-standard characters
        formula = formula.replace('-', '')

        # Regular expression to match elements and their counts
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)

        # Sum up elemental counts
        for element, count in matches:
            if element in element_counts:
                element_counts[element] += float(count) if count else 1.0
            else:
                element_counts[element] = float(count) if count else 1.0

        return element_counts

    def _extract_features(self, df):
        """
        Extract features from chemical formulas and structure types
        """
        # Find all unique elements in the dataset
        all_elements = set()
        for formula in df['element'].dropna():
            elements = self._parse_formula(formula).keys()
            all_elements.update(elements)

        # Filter the element list to only include elements found in the dataset
        used_elements = [elem for elem in self.elements if elem in all_elements]
        print(f"Found {len(used_elements)} unique elements in the dataset")

        # Create feature matrix for elements
        X_elements = np.zeros((len(df), len(used_elements)))

        for i, formula in enumerate(df['element']):
            if pd.isna(formula):
                continue

            element_counts = self._parse_formula(formula)
            total_count = sum(element_counts.values())

            if total_count > 0:  # Avoid division by zero
                for j, element in enumerate(used_elements):
                    if element in element_counts:
                        # Normalize by total count to get atomic proportions
                        X_elements[i, j] = element_counts[element] / total_count

        # Add structure type as one-hot encoding if available
        if 'str3' in df.columns:
            # Get all unique structure types (excluding NaN values)
            if not self.structure_types:
                # Convert to string and handle NaN values
                valid_structs = df['str3'].dropna().astype(str).unique().tolist()
                self.structure_types = sorted(valid_structs)
                print(f"Found {len(self.structure_types)} unique structure types")

            # Create one-hot encoded columns for each structure type
            X_struct = np.zeros((len(df), len(self.structure_types)))

            for i, struct in enumerate(df['str3']):
                if pd.isna(struct):
                    continue

                struct_str = str(struct)
                if struct_str in self.structure_types:
                    idx = self.structure_types.index(struct_str)
                    X_struct[i, idx] = 1

            # Combine element features with structure features
            X = np.hstack((X_elements, X_struct))
            feature_columns = used_elements + [f'structure_{st}' for st in self.structure_types]
        else:
            X = X_elements
            feature_columns = used_elements

        self.feature_columns = feature_columns
        return X

    def load_and_process_data(self, tsv_path, test_size=0.2, random_state=42):
        """
        Load and process the superconductor dataset from TSV format
        """
        # Load data with tab separator
        df = pd.read_csv(tsv_path, sep='\t')

        print(f"Loaded dataset with columns: {df.columns.tolist()}")
        print(f"Dataset shape: {df.shape}")

        # Check for missing values
        if df.isnull().any().any():
            print(f"Warning: Dataset contains {df.isnull().sum().sum()} missing values")
            # Drop rows with missing values in essential columns
            df = df.dropna(subset=['element', 'tc'])
            print(f"After dropping rows with missing element or tc: {df.shape}")

        # Extract features from chemical formula and structure
        X = self._extract_features(df)
        y = df['tc'].values

        print(f"Extracted feature matrix shape: {X.shape}")
        print(f"Number of feature columns: {len(self.feature_columns)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler and feature information
        if not os.path.exists('./models'):
            os.makedirs('./models')

        # Save everything needed for consistent inference
        with open('./models/supercon_processor.pkl', 'wb') as f:
            processor_data = {
                'scaler': self.scaler,
                'elements': self.elements,
                'used_elements': self.feature_columns[:-(len(self.structure_types) if self.structure_types else 0)],
                'structure_types': self.structure_types,
                'feature_columns': self.feature_columns,
                'n_features': X.shape[1]
            }
            pickle.dump(processor_data, f)

        print(f"Processed {len(y_train)} training samples and {len(y_test)} test samples")
        print(f"Feature dimension: {X_train_scaled.shape[1]}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def process_input(self, input_data):
        """
        Process input data for inference
        """
        # Load saved processor data
        try:
            with open('./models/supercon_processor.pkl', 'rb') as f:
                processor_data = pickle.load(f)
                self.scaler = processor_data['scaler']
                self.elements = processor_data['elements']
                used_elements = processor_data['used_elements']
                self.structure_types = processor_data['structure_types']
                self.feature_columns = processor_data['feature_columns']
                expected_features = processor_data['n_features']

            print(f"Loaded processor with {len(self.feature_columns)} features")
        except FileNotFoundError:
            raise FileNotFoundError("Processor data not found. Please train the model first.")

        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Check required columns
        if 'element' not in input_data.columns:
            raise ValueError("Missing required field: 'element' (chemical formula)")

        # Create feature matrix matching the training data format
        num_samples = len(input_data)
        X = np.zeros((num_samples, expected_features))

        for i, formula in enumerate(input_data['element']):
            # Process chemical formula
            element_counts = self._parse_formula(formula)
            total_count = sum(element_counts.values())

            if total_count > 0:  # Avoid division by zero
                # Fill element features
                for j, element in enumerate(used_elements):
                    if element in element_counts:
                        X[i, j] = element_counts[element] / total_count

            # Fill structure type features if available
            if 'str3' in input_data.columns and self.structure_types:
                struct = input_data.iloc[i]['str3']
                if not pd.isna(struct):
                    struct_str = str(struct)
                    if struct_str in self.structure_types:
                        idx = self.structure_types.index(struct_str)
                        X[i, len(used_elements) + idx] = 1

        # Scale features
        scaled_data = self.scaler.transform(X)
        return scaled_data