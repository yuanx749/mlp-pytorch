import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        print('{} seconds'.format(time.time() - start))
        return value
    return wrapper

class MLP(nn.Module):
    def __init__(self, D_in, H_sizes, D_out):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(D_in, H_sizes[0])])
        for i in range(len(H_sizes) - 1):
            self.linear.append(nn.Linear(H_sizes[i], H_sizes[i + 1]))
        self.linear.append(nn.Linear(H_sizes[-1], D_out))
    
    def forward(self, x):
        for linear in self.linear[:-1]:
            x = F.relu(linear(x))
        y_pred = self.linear[-1](x)
        return y_pred

class MLPClassifier:
    def __init__(
        self, hidden_layer_sizes=(100,), alpha=0.0001, 
        batch_size=200, learning_rate=0.001, max_iter=200, 
        shuffle=True, random_state=None, verbose=False):
        self.model = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.loss_curve = []

    @timer
    def fit(self, X, y):
        '''
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        '''
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size
        # n_classes = int(np.max(y)) + 1
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        dataset = TensorDataset(X, y)
        self.batch_size = min(self.batch_size, n_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.model = MLP(n_features, self.hidden_layer_sizes, n_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)

        self.model.train()
        for epoch in range(self.max_iter):
            accumulated_loss = 0.0
            for X, y in dataloader:
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                accumulated_loss += loss.item() * y.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_ = accumulated_loss / n_samples
            self.loss_curve.append(loss_)
            if self.verbose and epoch % 100 == 0:
                print(epoch, loss_)
        return self

    def predict(self, X):
        X = torch.from_numpy(X).float()
        self.model.eval()
        output = self.model(X)
        y_pred = torch.argmax(output, dim=1).numpy()
        return y_pred

    def predict_proba(self, X):
        X = torch.from_numpy(X).float()
        self.model.eval()
        output = self.model(X)
        y_prob = F.softmax(output, dim=1).detach().numpy()
        return y_prob

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / y.size
