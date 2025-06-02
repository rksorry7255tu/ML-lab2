import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=1000):
        self.k = k
        self.max_iters = max_iters

    def init_centroids(self, X):
        np.random.seed(0)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        centroids = self.init_centroids(X)
        for _ in range(self.max_iters):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return centroids, labels

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Apply KMeans
    kmeans = KMeans(k=2)
    centroids, labels = kmeans.fit(X)

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
