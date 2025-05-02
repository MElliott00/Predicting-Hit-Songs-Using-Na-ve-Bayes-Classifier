"""
naive_bayes_model.py

This module provides a Naive Bayes classifier for continuous features.
It includes:
  - CountingProbDist: A class to count observations and estimate probabilities.
  - NaiveBayesContinuous: A function that trains the classifier and returns a prediction function.
  
Reference:
The NaiveBayesContinuous model is based on our course’s GitHub repository 
(https://github.com/aimacode/aima-python) and the material in Chapter 20.
"""

import heapq
from src.dataset_utils import weighted_sampler, gaussian

class CountingProbDist:
    """
    A probability distribution based on counting observations.
    
    Methods:
        add(o): Increment count for observation o.
        __getitem__(item): Returns the probability of item.
    """
    def __init__(self, observations=None, default=0):
        if observations is None:
            observations = []
        self.dictionary = {}
        self.n_obs = 0
        self.default = default
        self.sampler = None

        for o in observations:
            self.add(o)

    def add(self, o):
        """Increment the count for the observation o."""
        self.smooth_for(o)
        self.dictionary[o] += 1
        self.n_obs += 1
        self.sampler = None

    def smooth_for(self, o):
        """Ensure the observation o is present with at least a default count."""
        if o not in self.dictionary:
            self.dictionary[o] = self.default
            self.n_obs += self.default
            self.sampler = None

    def __getitem__(self, item):
        """Return the probability of the item."""
        self.smooth_for(item)
        return self.dictionary[item] / self.n_obs

    def top(self, n):
        """Return the top n most frequent observations."""
        return heapq.nlargest(n, [(v, k) for (k, v) in self.dictionary.items()])

    def sample(self):
        """Return a random sample from the distribution."""
        if self.sampler is None:
            self.sampler = weighted_sampler(list(self.dictionary.keys()), list(self.dictionary.values()))
        return self.sampler()

def NaiveBayesContinuous(dataset):
    """
    Trains a Naive Bayes classifier using continuous features.
    
    Parameters:
        dataset (DataSet): A DataSet object containing the training data.
    
    Returns:
        predict (function): A function that takes a feature dictionary and returns a predicted target value.
    """
    means, deviations = dataset.find_means_and_deviations()
    target_vals = dataset.values[dataset.target]
    target_dist = CountingProbDist(target_vals)

    def predict(example):
        """Predict the target value for a given example."""
        def class_probability(target_val):
            prob = target_dist[target_val]
            for attr in dataset.inputs:
                prob *= gaussian(means[target_val][attr], deviations[target_val][attr], example[attr])
            return prob
        return max(target_vals, key=class_probability)

    return predict
