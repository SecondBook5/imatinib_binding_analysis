import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self):
        pass

    def read_data(self, file_path):
        """
        Read data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at '{
                                    file_path}'. Please provide a valid file path.")
        except Exception as e:
            raise Exception(
                f"An error occurred while reading the file: {str(e)}")

    def clean_data(self, data):
        """
        Clean and preprocess the raw data.

        Args:
            data (pd.DataFrame): Raw data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        # Handle missing values
        cleaned_data = data.dropna()

        # Remove outliers using z-score method
        z_scores = np.abs(
            (cleaned_data - cleaned_data.mean()) / cleaned_data.std())
        cleaned_data = cleaned_data[(z_scores < 3).all(axis=1)]

        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()

        # Validate specific columns for impossible cases
        if 'time' in cleaned_data.columns:
            if (cleaned_data['time'] < 0).any():
                raise ValueError("Time values cannot be negative.")
        if 'fraction_bound' in cleaned_data.columns:
            if (cleaned_data['fraction_bound'] < 0).any() or (cleaned_data['fraction_bound'] > 1).any():
                raise ValueError(
                    "Fraction bound values must be between 0 and 1.")
        if 'fluorescence' in cleaned_data.columns:
            if (cleaned_data['fluorescence'] < 0).any():
                raise ValueError("Fluorescence values cannot be negative.")

        if cleaned_data.empty:
            raise ValueError("No data available after cleaning.")

        return cleaned_data

    def normalize_data(self, data):
        """
        Normalize the data.

        Args:
            data (pd.DataFrame): Data to be normalized.

        Returns:
            pd.DataFrame: Normalized data.
        """
        # Perform standardization using z-scores
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)

        return pd.DataFrame(normalized_data, columns=data.columns)

    def organize_data(self, data):
        """
        Organize the data into appropriate data structures.

        Args:
            data (pd.DataFrame): Data.

        Returns:
            dict: Organized data.
        """
        # Organize data into dictionary format
        organized_data = {}
        for column in data.columns:
            organized_data[column] = data[column].values

        return organized_data

    def split_data(self, data, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            data (pd.DataFrame): Data to be split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Tuple containing training and testing sets.
        """
        if not 0 < test_size <= 1:
            raise ValueError("test_size must be in the range (0, 1].")

        # Split data into training and testing sets
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state)

        return train_data, test_data
