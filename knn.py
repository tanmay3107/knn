# knn.py
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """Calculates the straight-line distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        """
        Initializes the K-Nearest Neighbors classifier.
        :param k: The number of nearest neighbors to poll for a vote.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        KNN is a 'lazy learner'. It doesn't optimize weights; 
        it just memorizes the dataset.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for a set of new data points.
        """
        # We must predict the label for each row in X
        predicted_labels = [self._predict_single(x) for x in X]
        return np.array(predicted_labels)

    def _predict_single(self, x):
        """
        Helper function to predict the class for a single data point.
        """
        # 1. Compute the distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # 2. Get the indices of the 'k' closest training data points
        # np.argsort returns the indices that would sort the array
        k_indices = np.argsort(distances)[:self.k]

        # 3. Extract the actual labels of those 'k' nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Perform a majority vote (find the most common class label)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]

# --- Quick Test ---
if __name__ == "__main__":
    # Fake classification dataset (e.g., predicting if a student passes based on hours studied and slept)
    X = np.array([
        [2, 4], [1, 3], [2, 3], # Class 0 (Failed)
        [8, 9], [7, 8], [9, 8]  # Class 1 (Passed)
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    print("🧠 Booting up KNN Classifier...")
    classifier = KNN(k=3)
    classifier.fit(X, y)

    # Let's test a new data point right in the middle
    X_test = np.array([[5, 5], [1, 2], [8, 8]])
    predictions = classifier.predict(X_test)

    print(f"🎯 Predictions for test data: {predictions}")
    print("Expected roughly: [0 or 1 depending on distance, 0, 1]")