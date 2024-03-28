# atp_data_handling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_atp_binding_data(file_path):
    """
    Read equilibrium binding data of ATP from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the equilibrium binding data.

    Returns:
        pd.DataFrame: DataFrame containing the equilibrium binding data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs while reading the file.
    """
    print("Reading ATP binding data...")
    try:
        data = pd.read_csv(file_path)
        print("Data read successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at '{
                                file_path}'. Please provide a valid file path.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")

    # Validate data format
    if 'free_[ATP](nm)' not in data.columns or 'fluorescence' not in data.columns:
        raise ValueError(
            "Invalid data format. Columns 'free_[ATP](nm)' and 'fluorescence' are required.")

    return data


def clean_atp_binding_data(data):
    """
    Clean and preprocess the raw equilibrium binding data.

    Args:
        data (pd.DataFrame): Raw equilibrium binding data.

    Returns:
        pd.DataFrame: Cleaned equilibrium binding data.
        
    Raises:
        ValueError: If data is empty after cleaning.
    """
    print("Cleaning ATP binding data...")
    # Handle missing values
    cleaned_data = data.dropna()
    print("Missing values handled.")

    # Remove outliers using z-score method
    z_scores = np.abs(
        (cleaned_data - cleaned_data.mean()) / cleaned_data.std())
    cleaned_data = cleaned_data[(z_scores < 3).all(axis=1)]
    print("Outliers removed.")

    # Remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()
    print("Duplicate rows removed.")

    if cleaned_data.empty:
        raise ValueError("No data available after cleaning.")

    return cleaned_data


def normalize_atp_binding_data(data):
    """
    Normalize the equilibrium binding data.

    Args:
        data (pd.DataFrame): Equilibrium binding data to be normalized.

    Returns:
        pd.DataFrame: Normalized equilibrium binding data.
    """
    print("Normalizing ATP binding data...")
    # Perform standardization using z-scores
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    print("Data normalized.")

    return pd.DataFrame(normalized_data, columns=data.columns)


def organize_atp_binding_data(data):
    """
    Organize the equilibrium binding data into appropriate data structures.

    Args:
        data (pd.DataFrame): Equilibrium binding data.

    Returns:
        dict: Organized equilibrium binding data.
    """
    print("Organizing ATP binding data...")
    # Organize data into dictionary format
    organized_data = {
        'atp_concentration': data['free_[ATP](nm)'].values,
        'fluorescence': data['fluorescence'].values
    }
    print("Data organized.")
    return organized_data


def split_atp_binding_data(data, test_size=0.2, random_state=42):
    """
    Split the equilibrium binding data into training and testing sets.

    Args:
        data (pd.DataFrame): Equilibrium binding data to be split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing training and testing sets.
        
    Raises:
        ValueError: If test_size is not in the range (0, 1].
    """
    print("Splitting ATP binding data...")
    if not 0 < test_size <= 1:
        raise ValueError("test_size must be in the range (0, 1].")

    # Split data into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state)
    
    print("Data split complete.")
    return train_data, test_data


