import time
from abc import ABCMeta, abstractmethod

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
        x = self.linear[-1](x)
        return x

class CNN(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        f = 5
        self.conv1 = nn.Conv1d(1, 6, f)
        self.pool = nn.MaxPool1d(2)
        l_out = self._output_size(D_in, f, 1, 0)
        l_out = self._output_size(l_out, 2, 2, 0)
        self.conv2 = nn.Conv1d(6, 16, f)
        l_out = self._output_size(l_out, f, 1, 0)
        l_out = self._output_size(l_out, 2, 2, 0)
        self.l_out = l_out
        self.fc1 = nn.Linear(16 * l_out, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, D_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.l_out)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _output_size(self, l_in, kernel_size, stride, padding):
        return int(np.floor((l_in+2*padding-kernel_size)/stride + 1))

class RNN(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, D_out)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

class BaseClassifier(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self, alpha, batch_size, learning_rate, 
        max_iter, shuffle, random_state, verbose, 
        validation_fraction):
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.validation_fraction = validation_fraction

    def _numpy_to_tensor_x(self, X):
        return torch.from_numpy(X).float()

    def _fetch_dataloader(self, X, y):
        X, self.X_val, y, self.y_val = train_test_split(
            X, y, 
            test_size=self.validation_fraction, 
            random_state=self.random_state, 
            stratify=y)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_samples, self.n_features = X.shape
        self.n_classes = np.unique(y).size

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)
        X = self._numpy_to_tensor_x(X)
        y = torch.from_numpy(y).long()
        dataset = TensorDataset(X, y)
        self.batch_size = min(self.batch_size, self.n_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def _train(self, dataloader):
        self.loss_curve_ = []
        self.training_scores_ = []
        self.validation_scores_ = []
        self.fit_times_ = []
        
        loss_fn = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.alpha)

        if self.verbose:
            print('{:>8}{:>8}{:>18}{:>18}{:>18}'.format('epoch', 'loss', 'training score', 'validation score', 'time (seconds)'))
        for epoch in range(self.max_iter):
            start = time.time()
            self.model.train()
            accumulated_loss = 0.0
            for X, y in dataloader:
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                accumulated_loss += loss.item() * y.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_ = accumulated_loss / self.n_samples
            self.loss_curve_.append(loss_)
            self.training_scores_.append(self.score(self.X_, self.y_))
            self.validation_scores_.append(self.score(self.X_val, self.y_val))
            self.fit_times_.append(time.time() - start)
            if self.verbose and epoch % 10 == 0:
                print('{:8d}{:8.4f}{:18.4f}{:18.4f}{:18.4f}'.format(
                    epoch + 1, loss_, self.training_scores_[-1], self.validation_scores_[-1], self.fit_times_[-1]))
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = self._numpy_to_tensor_x(X)
        self.model.eval()
        output = self.model(X)
        y_pred = self.classes_[torch.argmax(output, dim=1).numpy()]
        return y_pred

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = self._numpy_to_tensor_x(X)
        self.model.eval()
        output = self.model(X)
        y_prob = F.softmax(output, dim=1).detach().numpy()
        return y_prob

class MLPClassifier(BaseClassifier):
    def __init__(
        self, hidden_layer_sizes=(100, 100), alpha=0.0001, 
        batch_size=200, learning_rate=0.001, max_iter=200, 
        shuffle=True, random_state=None, verbose=False, 
        validation_fraction=0.2):
        # no *args or **kwargs
        super().__init__(
            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, 
            max_iter=max_iter, shuffle=shuffle, random_state=random_state, verbose=verbose, 
            validation_fraction=validation_fraction)
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, y):
        '''
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        '''
        dataloader = self._fetch_dataloader(X, y)
        self.model = MLP(self.n_features, self.hidden_layer_sizes, self.n_classes)
        return self._train(dataloader)

class CNNClassifier(BaseClassifier):
    def __init__(
        self, alpha=0.0001, 
        batch_size=200, learning_rate=0.001, max_iter=200, 
        shuffle=True, random_state=None, verbose=False, 
        validation_fraction=0.2):
        super().__init__(
            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, 
            max_iter=max_iter, shuffle=shuffle, random_state=random_state, verbose=verbose, 
            validation_fraction=validation_fraction)

    def _numpy_to_tensor_x(self, X):
        return torch.from_numpy(X).float().unsqueeze(1) # input shape: (N, C_in, L_in)

    def fit(self, X, y):
        '''
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        '''
        dataloader = self._fetch_dataloader(X, y)
        self.model = CNN(self.n_features, self.n_classes)
        return self._train(dataloader)

class RNNClassifier(BaseClassifier):
    def __init__(
        self, alpha=0.0001, 
        batch_size=200, learning_rate=0.001, max_iter=200, 
        shuffle=True, random_state=None, verbose=False, 
        validation_fraction=0.2):
        super().__init__(
            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, 
            max_iter=max_iter, shuffle=shuffle, random_state=random_state, verbose=verbose, 
            validation_fraction=validation_fraction)

    def _numpy_to_tensor_x(self, X):
        return torch.from_numpy(X).float().unsqueeze(2) # input shape: (batch, seq_len, input_size)

    def fit(self, X, y):
        '''
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        '''
        dataloader = self._fetch_dataloader(X, y)
        self.model = RNN(self.n_features, self.n_classes)
        return self._train(dataloader)

if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(MLPClassifier())