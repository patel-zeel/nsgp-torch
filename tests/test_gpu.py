import pytest
import matplotlib.pyplot as plt
from time import time


def model(device):
    from nsgp import NSGP
    from nsgp.utils.inducing_functions import f_kmeans
    import torch
    import numpy as np

    X = torch.rand(1000, 3)*100
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1) + torch.rand(1000, 1)

    X_new = torch.rand(10000, 3)
    X_bar = f_kmeans(X, num_inducing_points=4, random_state=None).to(device)
    X = X.to(device)
    y = y.to(device)
    X_new = X_new.to(device)

    model = NSGP(X, y, X_bar=X_bar, jitter=10**-5)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []

    # Move it to device
    model.to(device)

    init = time()
    model.train()
    for _ in range(200):
        optim.zero_grad()
        loss = model()
        losses.append(loss.item())
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        y_new, y_var = model.predict(X_new)
        y_std2 = y_var.diagonal()**0.5 * 2
        print(time()-init, 'seconds')
        # fig, ax = plt.subplots(3, 1, figsize=(10, 16))
        # ax[0].scatter(X, y)
        # ax[0].plot(X_new.numpy(), y_new.numpy())
        # ax[0].fill_between(X_new.ravel(), y_new.ravel() -
        #                    y_std2, y_new.ravel()+y_std2, alpha=0.5)

        # ax[2].plot(losses)

        # ax[1].plot(X_new, model.get_LS(X_new)[0])

        # fig.savefig('./test_step_function.pdf')


def test_model():
    model('cuda')
    model('cpu')
