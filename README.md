# NSGP

### Example

```python
### Imports
from nsgp import NSGP
from nsgp.utils.inducing_functions import f_kmeans
import torch
import matplotlib.pyplot as plt

### Dataset
num_low=25
num_high=25
gap = -.1
noise=0.0001
x = torch.vstack((torch.linspace(-1, -gap/2.0, num_low).reshape(-1,1),
              torch.linspace(gap/2.0, 1, num_high).reshape(-1,1)))
x_new = torch.linspace(-1,1,100).reshape(-1,1)
y = torch.vstack((torch.zeros((num_low, 1)), torch.ones((num_high,1)))) + torch.rand(50,1)*0.1
scale = torch.sqrt(y.var())
offset = y.mean()
y = (y-offset)/scale

### Model definition
X_bar = f_kmeans(x, num_inducing_points=3)
model = NSGP(x, y, X_bar)

### Model training
torch.manual_seed(0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in range(100):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()

### Test
model.eval()
y_hat, y_var = model.predict(x_new)
y_std2 = torch.sqrt(y_var.diagonal())*2

### Plot
fig, ax = plt.subplots(1,2,figsize=(12,3.5))
with torch.no_grad():
    ax[0].scatter(x, y)
    ax[0].plot(x_new, y_hat)
    ax[0].fill_between(x_new.ravel(), y_hat.ravel()-y_std2, y_hat.ravel()+y_std2, alpha=0.5)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('y')
    ax[1].plot(x_new, model.get_LS(x_new, 0))
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Lengthscale')
```
