import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split

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

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, hidden_layer_sizes=(100,), alpha=0.0001, 
        batch_size=200, learning_rate=0.001, max_iter=200, 
        shuffle=True, random_state=None, verbose=False, 
        validation_fraction=0.1):
        self.model = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.validation_fraction = validation_fraction
        self.loss_curve_ = []
        self.training_scores_ = []
        self.validation_scores_ = []

    def timer(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            value = func(self, *args, **kwargs)
            if self.verbose:
                print('{} seconds'.format(time.time() - start))
            return value
        return wrapper
    
    @timer
    def fit(self, X, y):
        '''
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        '''
        X, X_val, y, y_val = train_test_split(
            X, y, 
            test_size=self.validation_fraction, 
            random_state=self.random_state, 
            stratify=y)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_, y = np.unique(y, return_inverse=True)
        
        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        dataset = TensorDataset(X, y)
        self.batch_size = min(self.batch_size, n_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.model = MLP(n_features, self.hidden_layer_sizes, n_classes)
        loss_fn = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.alpha)

        if self.verbose:
            print('{:>8}{:>8}{:>18}{:>18}'.format('epoch', 'loss', 'training score', 'validation score'))
        for epoch in range(self.max_iter):
            self.model.train()
            accumulated_loss = 0.0
            for X, y in dataloader:
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                accumulated_loss += loss.item() * y.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_ = accumulated_loss / n_samples
            self.loss_curve_.append(loss_)
            self.training_scores_.append(self.score(self.X_, self.y_))
            self.validation_scores_.append(self.score(X_val, y_val))
            if self.verbose and epoch % 10 == 0:
                print('{:8d}{:8.4f}{:18.4f}{:18.4f}'.format(
                    epoch + 1, loss_, self.training_scores_[-1], self.validation_scores_[-1]))
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = torch.from_numpy(X).float()
        self.model.eval()
        output = self.model(X)
        y_pred = self.classes_[torch.argmax(output, dim=1).numpy()]
        return y_pred

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = torch.from_numpy(X).float()
        self.model.eval()
        output = self.model(X)
        y_prob = F.softmax(output, dim=1).detach().numpy()
        return y_prob
