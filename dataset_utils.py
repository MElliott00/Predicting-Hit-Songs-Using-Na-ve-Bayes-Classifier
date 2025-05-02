"""
dataset_utils.py

This module contains helper functions and a simple DataSet class used for the
Naive Bayes assignment. It provides functions to:
  - Load a CSV dataset into a pandas DataFrame.
  - Split the dataset into training and testing sets.
  - Compute the Gaussian probability.
  - Wrap a DataFrame in a DataSet class with necessary attributes and methods.
"""

import bisect
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def weighted_sampler(seq, weights):
    """
    Returns a function that, when called, returns a random element from seq.
    The elements are chosen based on the provided weights.
    """
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def load_dataset(path):
    """
    Loads a CSV file from the specified path into a pandas DataFrame.
    """
    return pd.read_csv(path)

def split_dataset(df, target, test_size=0.2):
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
        df (DataFrame): The dataset.
        target (str): The target column name.
        test_size (float): Fraction of data to be used as the test set.
    
    Returns:
        (train_df, test_df): Two DataFrames with reset indices.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def gaussian(mean, st_dev, x):
    """
    Calculates the Gaussian probability density of x given the mean and standard deviation.
    
    Parameters:
        mean (float): The mean of the distribution.
        st_dev (float): The standard deviation of the distribution.
        x (float): The value to evaluate.
    
    Returns:
        float: The probability density of x.
    """
    return 1 / (np.sqrt(2 * np.pi) * st_dev) * np.e ** (-0.5 * ((float(x - mean) / st_dev) ** 2))

class DataSet:
    """
    A simple wrapper for a pandas DataFrame that provides the attributes required 
    by the NaiveBayesContinuous model.
    
    Attributes:
        df (DataFrame): The underlying pandas DataFrame.
        target (str): The target column name.
        inputs (list): List of feature column names (all columns except target).
        values (dict): A dictionary with keys as column names and values as lists of column values.
    
    Methods:
        find_means_and_deviations(): Computes means and standard deviations for each 
        input feature grouped by the target values.

    Usage:
        dataset = DataSet(train_df, 'target_column_name')
        means, deviations = dataset.find_means_and_deviations()
    """
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.inputs = [col for col in df.columns if col != target]
        self.values = {col: df[col].tolist() for col in df.columns}
    
    def find_means_and_deviations(self):
        """
        Computes the mean and standard deviation for each input feature, grouped by target value.
        
        Returns:
            means (dict): Nested dictionary of means {target: {feature: mean}}.
            deviations (dict): Nested dictionary of standard deviations {target: {feature: std}}.
        """
        means = {}
        deviations = {}
        unique_targets = self.df[self.target].unique()
        for t in unique_targets:
            df_t = self.df[self.df[self.target] == t]
            means[t] = {}
            deviations[t] = {}
            for col in self.inputs:
                means[t][col] = df_t[col].mean()
                std = df_t[col].std(ddof=0)
                deviations[t][col] = std if std != 0 else 1e-6  # Avoid division by zero.
        return means, deviations
