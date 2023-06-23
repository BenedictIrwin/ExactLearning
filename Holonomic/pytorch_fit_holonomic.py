import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy

# n = 1000
# noise = torch.Tensor(np.random.normal(0, 0.02, size=n))

class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self):
        
        super().__init__()

        num_derivs = 3
        num_powers = 3

        # initialize weights with random numbers
        weights = torch.distributions.Uniform(-10, 10).sample((num_derivs,num_powers))
        
        if(False):
          weights[0,0]= -25
          weights[0,1]= 0
          weights[0,2]= 1
          weights[1,0]= 0
          weights[1,1]= 1
          weights[1,2]= 0
          weights[2,0]= 0
          weights[2,1]= 0
          weights[2,2]= 1
       
        weights=weights.double() 

        # make weights torch parameters
        self.weights = nn.Parameter(weights)        
        
    def forward(self, X, Y):
        """
        Holonomic of the form [y,y',y'',...].weights.[x,x**2,x**3,...] = 0
        """
        ret = torch.einsum("it,ij->jt",Y,self.weights)
        ret = torch.einsum("jt,jt->t",ret,X)
        return ret
    
def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        preds = model(X,Y)
	# Want the smallest max element (i.e.) fluctuation
        loss = torch.mean(preds**2) + torch.max(torch.abs(preds))**2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        print(loss)  
    return losses

# Load up the generated data of f, f', f''

# Eventually, we would add a dimension to tensor (gradient order)
x = np.load('x.npy')
y = np.load('y.npy')
dy = np.load('dy.npy')
ddy = np.load('ddy.npy')

# Convert them to constant tensors
x = torch.from_numpy(x)
y = torch.from_numpy(y)
dy = torch.from_numpy(dy)
ddy = torch.from_numpy(ddy)

Y = torch.stack([y,dy,ddy])
print(Y.shape)

X = torch.stack([x**0,x,x**2])
print(X.shape)

# instantiate model
m = Model()

print("start ",m.weights)
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.1)
losses = training_loop(m, opt, n=10000)

preds = m(X,Y)

final_weights = m.weights.detach().numpy()
print(torch.mean(torch.abs(preds)))
print(final_weights)
print(np.round(final_weights,2))

preds = preds.detach().numpy()
plt.plot(range(len(preds)),preds)
plt.show()

