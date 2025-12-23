import os
import pickle
import math
import time
import torch
import torch.nn.functional
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = "/kaggle/input/fii-atnn-2025-competition-1", train: bool = True):
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]

def to_tensor_dataset(dataset):
    X_list = []
    Y_list = []
    for image, label in dataset:
        arr = np.asarray(image, dtype=np.float32)
        arr = arr.reshape(-1)
        X_list.append(arr)
        if label is not None:
            Y_list.append(int(label))
    X = np.stack(X_list, axis=0)
    X = X / 255.0
    X = torch.from_numpy(X).float()
    if len(Y_list) > 0:
        Y = torch.tensor(Y_list, dtype=torch.long)
    else:
        Y = None
    return X, Y

def accuracy(pred_logits, targets):
    pred = pred_logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def one_hot(labels, num_classes):
    y = torch.zeros(labels.size(0), num_classes, device=labels.device)
    y.scatter_(1, labels.unsqueeze(1), 1.0)
    return y

class MLP:
    def __init__(self, input_dim=784, hidden=100, output_dim=10, device='cpu', seed=42):
        self.device = device
        torch.manual_seed(seed)
        def xavier(in_dim, out_dim):
            std = math.sqrt(2.0 / (in_dim + out_dim))
            return torch.randn(out_dim, in_dim, device=self.device) * std

        self.W1 = xavier(input_dim, hidden).requires_grad_(False)
        self.b1 = torch.zeros(hidden, device=self.device).requires_grad_(False)
        self.W2 = xavier(hidden, output_dim).requires_grad_(False)
        self.b2 = torch.zeros(output_dim, device=self.device).requires_grad_(False)

        self.vW1 = torch.zeros_like(self.W1)
        self.vb1 = torch.zeros_like(self.b1)
        self.vW2 = torch.zeros_like(self.W2)
        self.vb2 = torch.zeros_like(self.b2)

    def forward(self, X):
        z1 = X.matmul(self.W1.t()) + self.b1.unsqueeze(0)
        a1 = torch.relu(z1)
        logits = a1.matmul(self.W2.t()) + self.b2.unsqueeze(0)
        return z1, a1, logits

def train(model, X_train, Y_train, X_val=None, Y_val=None,
                 epochs=20, batch_size=128, lr=0.1, momentum=0.9, weight_decay=0.0,
                 device='cpu', verbose=True):
    N = X_train.shape[0]
    classes = model.b2.shape[0]
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    if X_val is not None:
        X_val = X_val.to(device)
        Y_val = Y_val.to(device)

    for epoch in range(1, epochs+1):
        model.W1 = model.W1.to(device)
        model.b1 = model.b1.to(device)
        model.W2 = model.W2.to(device)
        model.b2 = model.b2.to(device)
        model.vW1 = model.vW1.to(device)
        model.vb1 = model.vb1.to(device)
        model.vW2 = model.vW2.to(device)
        model.vb2 = model.vb2.to(device)

        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            batches += 1
            idx = perm[i:i+batch_size]
            Xb = X_train[idx]
            Yb = Y_train[idx]
            B = Xb.shape[0]

            z1, a1, logits = model.forward(Xb)
            loss_val = torch.nn.functional.cross_entropy(logits, Yb, reduction='mean')
            epoch_loss += loss_val.item() * B

            epoch_acc += accuracy(logits, Yb) * B

            logits_max = logits.max(dim=1, keepdim=True)[0]
            exp_logits = (logits - logits_max).exp()
            probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

            y_onehot = one_hot(Yb, classes)

            dlogits = (probs - y_onehot) / B
            dW2 = dlogits.t().matmul(a1)
            db2 = dlogits.sum(dim=0)

            da1 = dlogits.matmul(model.W2)
            dz1 = da1 * (z1 > 0).float()

            dW1 = dz1.t().matmul(Xb)
            db1 = dz1.sum(dim=0)

            if weight_decay != 0:
                dW2 = dW2 + weight_decay * model.W2
                dW1 = dW1 + weight_decay * model.W1

            model.vW2 = momentum * model.vW2 - lr * dW2
            model.vb2 = momentum * model.vb2 - lr * db2
            model.W2 = model.W2 + model.vW2
            model.b2 = model.b2 + model.vb2

            model.vW1 = momentum * model.vW1 - lr * dW1
            model.vb1 = momentum * model.vb1 - lr * db1
            model.W1 = model.W1 + model.vW1
            model.b1 = model.b1 + model.vb1

        epoch_loss = epoch_loss / N
        epoch_acc = epoch_acc / N

        val_loss = None
        val_acc = None
        if X_val is not None:
            with torch.no_grad():
                _, _, val_logits = model.forward(X_val)
                val_loss = torch.nn.functional.cross_entropy(val_logits, Y_val, reduction='mean').item()
                val_acc = accuracy(val_logits, Y_val)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            t1 = time.time()
            print(f"Epoch {epoch:2d} | time {t1-t0:4.2f}s | train_loss {epoch_loss:.4f} | train_acc {epoch_acc*100:.2f}%"
                  + (f" | val_loss {val_loss:.4f} | val_acc {val_acc*100:.2f}%" if val_loss is not None else ""))

    return history

if __name__ == "__main__":
    root = "/kaggle/input/fii-atnn-2025-competition-1"
    train_dataset = ExtendedMNISTDataset(root=root, train=True)
    test_dataset = ExtendedMNISTDataset(root=root, train=False)

    X_all, Y_all = to_tensor_dataset(train_dataset)
    X_test, _ = to_tensor_dataset(test_dataset)

    N = X_all.shape[0]
    val_split = int(0.1 * N)
    perm = torch.randperm(N)
    val_idx = perm[:val_split]
    train_idx = perm[val_split:]

    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_val = X_all[val_idx]
    Y_val = Y_all[val_idx]

    device = "cuda"
    print("Using device:", device)
    model = MLP(input_dim=784, hidden=100, output_dim=10, device=device, seed=2025)

    epochs = 200
    batch_size = 128
    lr = 0.11
    momentum = 0.89
    weight_decay = 0.00001

    hist = train(model, X_train, Y_train, X_val, Y_val,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        momentum=momentum, weight_decay=weight_decay,
                        device=device, verbose=True)

    with torch.no_grad():
        _, _, val_logits = model.forward(X_val.to(device))
        val_loss = torch.nn.functional.cross_entropy(val_logits, Y_val.to(device), reduction='mean').item()
        val_acc = accuracy(val_logits, Y_val.to(device))
    print(f"Final val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

    model.W1 = model.W1.to(device)
    model.b1 = model.b1.to(device)
    model.W2 = model.W2.to(device)
    model.b2 = model.b2.to(device)
    X_test = X_test.to(device)

    all_preds = []
    batch_size_inf = 512
    for i in range(0, X_test.shape[0], batch_size_inf):
        Xb = X_test[i:i+batch_size_inf]
        _, _, logits = model.forward(Xb)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)

    submission = {
        "ID": list(range(len(all_preds))),
        "target": all_preds
    }
    df = pd.DataFrame(submission)
    df.to_csv("submission.csv", index=False)
    print("Saved submission.csv with", len(all_preds), "rows.")
