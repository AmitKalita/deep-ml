import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values

    X_mean = np.nanmean(X, axis=0)
    X_nan = np.isnan(X)

    X_clean = np.where(X_nan, X_mean, X)

    
    return X_clean
