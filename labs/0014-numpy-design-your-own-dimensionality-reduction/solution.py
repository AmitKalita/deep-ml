import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        self.X_mean = self
        self.proj = self 

        # Add any attributes you need to store learned parameters
    
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Analyze X and store what you need for transform()
        self.X_mean = np.mean(X, axis=0)
        X_cent = X - self.X_mean
        n = X.shape[0]

        Cov_mat = 1/n * (X_cent. T @ X_cent)

        eigenvalues, eigenvectors = np.linalg.eigh(Cov_mat)

        sorted_eigenvalues_ind = np.argsort(eigenvalues)[::-1][:self.n_components]

        project_mat = np.array([])
        for ind in sorted_eigenvalues_ind:
            project_mat=np.append(project_mat, eigenvectors[ind])
        
        self.proj = project_mat.reshape(self.n_components,-1).T

        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Project X to 10 dimensions using parameters from fit()
        X_cent = X - self.X_mean
        X_reduce = X_cent @ self.proj
        return X_reduce
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
