import numpy as np
import torch
from utils.nn_CLSTM import x_LSTM

if __name__ == "__main__":
    obj = lambda x: x**2
    jac = lambda x: 2*x
    # add constraints x < 2
    r = 0
    z = 3
    gamma = 0
    rho = 1
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    x_model = x_LSTM(1,arg_nn)
    x_optimizers = torch.optim.Adam(x_model.parameters(), lr=0.004)
    z = lambda x: np.exp(-x) + 2 + 1 * np.sin(x)  # Example function for z(k)
    
    x = torch.tensor([[100]])
    for k in range(1000):
        for j in range(10):
            r = np.maximum(r + 0.1 * (z(k)-2.5),0)
            grad_x = jac(x) + r + gamma + rho * (x - z(k))
            delta, _ = x_model(grad_x)
            _x = x + delta[0]
            x = _x.detach()
            grad_x = jac(x)+ r + gamma+ rho * (x - z(k))
            _x.backward(grad_x, retain_graph=True)
            gamma = gamma + rho * (x - z(k))

        x_optimizers.step()
        x_optimizers.zero_grad()
        print("x:",x,"lambda:",r)
