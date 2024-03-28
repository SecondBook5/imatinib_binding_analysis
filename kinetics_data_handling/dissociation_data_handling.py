# dissociation_data_handling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_dissociation_data(file_path):
    """
    Read dissociation data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the dissociation data.

    Returns:
        pd.DataFrame: DataFrame containing the dissociation data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs while reading the file.
    """
    print("Reading dissociation data...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at '{
                                file_path}'. Please provide a valid file path.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")

    # Validate data format
    if 'time' not in data.columns or 'fluorescence' not in data.columns:
        raise ValueError(
            "Invalid data format. Columns 'time' and 'fluorescence' are required.")
    
    print("Dissociation data read successfully.")
    return data


def clean_dissociation_data(data):
    """
    Clean and preprocess the raw dissociation data.

    Args:
        data (pd.DataFrame): Raw dissociation data.

    Returns:
        pd.DataFrame: Cleaned dissociation data.
        
    Raises:
        ValueError: If data is empty after cleaning.
    """
    print("Cleaning dissociation data...")
    # Handle missing values
    cleaned_data = data.dropna()

    # Remove outliers using z-score method
    z_scores = np.abs(
        (cleaned_data - cleaned_data.mean()) / cleaned_data.std())
    cleaned_data = cleaned_data[(z_scores < 3).all(axis=1)]

    # Remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()

    # Ensure time values are non-negative
    if (cleaned_data['time'] < 0).any():
        raise ValueError("Time values cannot be negative.")

    if cleaned_data.empty:
        raise ValueError("No data available after cleaning.")

    print("Dissociation data cleaned successfully.")
    return cleaned_data


def normalize_dissociation_data(data):
    """
    Normalize the dissociation data.

    Args:
        data (pd.DataFrame): Dissociation data to be normalized.

    Returns:
        pd.DataFrame: Normalized dissociation data.
    """
    print("Normalizing dissociation data...")
    # Perform standardization using z-scores
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    print("Dissociation data normalized successfully.")
    return pd.DataFrame(normalized_data, columns=data.columns)


def organize_dissociation_data(data):
    """
    Organize the dissociation data into appropriate data structures.

    Args:
        data (pd.DataFrame): Dissociation data.

    Returns:
        dict: Organized dissociation data.
    """
    print("Organizing dissociation data...")
    organized_data = {
        'time': data['time'].values,
        'fluorescence': data['fluorescence'].values
    }

    print("Dissociation data organized successfully.")
    return organized_data


def split_dissociation_data(data, test_size=0.2, random_state=42):
    """
    Split the dissociation data into training and testing sets.

    Args:
        data (pd.DataFrame): Dissociation data to be split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing training and testing sets.
        
    Raises:
        ValueError: If test_size is not in the range (0, 1].
    """

    print("Splitting dissociation data into training and testing sets...")
    if not 0 < test_size <= 1:
        raise ValueError("test_size must be in the range (0, 1].")

    # Split data into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state)
    
    print("Dissociation data split successfully.")
    return train_data, test_data
