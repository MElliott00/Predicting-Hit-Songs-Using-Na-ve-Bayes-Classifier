import pandas as pd
import numpy as np
from src.dataset_utils import split_dataset, DataSet
from src.naive_bayes_model import NaiveBayesContinuous

# 1. Load the raw CSV once
raw = pd.read_csv("data/spotify_hits.csv")

# 2. Choose the thresholds to test
thresholds = [65, 70, 75, 80]

# 3. Dictionary to hold accuracy for each threshold
results = {}

for T in thresholds:
    # a) Create a fresh copy of the data and define the binary target
    df = raw.copy()
    df['hit'] = (df['popularity'] >= T).astype(int)
    
    # b) Keep only numeric columns and drop any NaNs
    df = df.select_dtypes(include=[np.number]).dropna()
    
    # c) Split into train/test
    train_df, test_df = split_dataset(df, 'hit', test_size=0.2)
    
    # d) Train the Naive Bayes model
    ds    = DataSet(train_df, 'hit')
    model = NaiveBayesContinuous(ds)
    
    # e) Evaluate accuracy on the test set
    correct = 0
    for _, row in test_df.iterrows():
        feat = row.drop('hit').to_dict()
        if model(feat) == row['hit']:
            correct += 1
    accuracy = correct / len(test_df)
    
    # f) Store the result
    results[T] = accuracy

# 4. Find the best threshold by highest accuracy
best_threshold = max(results, key=results.get)
best_accuracy  = results[best_threshold]

# 5. Report
print("Threshold accuracies:", results)
print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy:.2f}")
