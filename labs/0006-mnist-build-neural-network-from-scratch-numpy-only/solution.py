import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        rng = np.random.default_rng(seed = 42)
        std = np.sqrt(2 / input_size)
        self.w1 = rng.normal(loc=0.0, scale=std, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))  
        self.w2 = rng.normal(loc=0.0, scale=std, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        pass
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        z1 = X @ self.w1 + self.b1  #N X 128
        h1 = np.maximum(0, z1)  
        z2 = h1 @ self.w2 + self.b2

        self.cache = {'X': X, 'z1': z1, 'h1': h1, 'z2': z2}

        max_z = np.max(z2, axis=1, keepdims=True)
        probs = np.exp(z2 - max_z)/np.sum(np.exp(z2 - max_z), axis=1, keepdims=True)

        return probs
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        # 2. Compute gradient of loss w.r.t. softmax output
        # 3. Backprop through linear layer 2
        # 4. Backprop through ReLU
        # 5. Backprop through linear layer 1
        # 6. Update all parameters using gradients
        
        N = X.shape[0]
        correct_probs = probs[np.arange(N), y]
        L = -np.mean(np.sum(np.log(correct_probs)))

        y_one_hot = np.zeros((probs.shape))
        y_one_hot[np.arange(X.shape[0]), y] = 1

        grad_loss_z2 = probs - y_one_hot #N X 10
        grad_loss_w2 = self.cache['h1'].T @ grad_loss_z2  #128 X 10
        grad_loss_b2 = np.sum(grad_loss_z2, axis=0, keepdims=True) #1 X 10
        grad_loss_h1 = grad_loss_z2 @ self.w2.T #N X 128

        grad_relu = (self.cache['z1'] > 0).astype(float) #N X 128
        grad_loss_z1 = grad_loss_h1 * grad_relu #N X 128
        grad_loss_w1 = X.T @ grad_loss_z1 #784 X 128
        grad_loss_b1 = np.sum(grad_loss_z1, axis=0, keepdims=True) #1 X 128

        self.w1 = self.w1 - self.lr * grad_loss_w1
        self.b1 = self.b1 - self.lr * grad_loss_b1
        self.w2 = self.w2 - self.lr * grad_loss_w2  
        self.b2 = self.b2 - self.lr * grad_loss_b2 

        return L

    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)