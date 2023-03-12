import numpy as np
from sklearn.utils import resample

def binaryUndersampling():
    # Define a dataset with imbalanced classes
    X = np.array([[0, 1], [1, 1], [0, 1], [1, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([0, 1, 0, 0, 0, 1, 1])

    # Separate the samples in each class
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]

    # Undersample the majority class
    n_samples0 = len(X_class0)
    n_samples1 = len(X_class1)

    n_samples = min(n_samples0, n_samples1)

    if(n_samples1 < n_samples0):
        X_class0 = resample(X_class0, n_samples=n_samples, replace=False)
    else:
        X_class1 = resample(X_class1, n_samples=n_samples, replace=False)

    # Combine the undersampled majority class with the minority class
    X_undersampled = np.vstack((X_class0, X_class1))
    y_undersampled = np.concatenate(([0] * n_samples, [1] * n_samples))

    # Print the balanced dataset
    print(X_undersampled)
    print(y_undersampled)

from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

# Create a synthetic dataset with imbalanced classes
X = np.array([[0, 1], [1, 1], [0, 1], [1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([0, 1, 0, 0, 0, 1, 1])

# Create an undersampler to balance the classes
undersampler = RandomUnderSampler(sampling_strategy='not minority')

# Resample the dataset
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Print the class distribution before and after resampling
print('Original dataset shape:', X.shape)
print('Original class distribution:', {i: sum(y == i) for i in set(y)})
print('Resampled dataset shape:', X_resampled.shape)
print('Resampled class distribution:', {i: sum(y_resampled == i) for i in set(y_resampled)})


pp = [[1,2], [2,3], [3,4]]
ppp = [4, 5, 6, [4, 5]]
pp.extend(ppp)

print(pp)
