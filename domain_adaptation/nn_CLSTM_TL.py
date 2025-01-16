import torch
import torch.nn as nn

# this is the centralized version without a projection layer 
class x_LSTM(nn.Module):
    # feed lambda and output x. Since centralized, we do not consider global consensus terms 
    def __init__(self,len_x,arg_nn) -> None:
        super(x_LSTM, self).__init__()
        self.len_x = len_x
        self.arg_nn = arg_nn
        self.net_x = nn.LSTM(input_size=len_x, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size, len_x)

    # r represents lambda
    def forward(self, dx, h_s = None):
        # Reshape x and lambda to match the input size of the LSTM
        dx = dx.view(-1, self.len_x)
        if h_s is None:
            out_temp = self.net_x(dx)
        else:
            out_temp = self.net_x(dx, h_s)
        out_x = out_temp[0]
        out_hidden = out_temp[1]
        
        out_x = self.net_fc(out_x)
        return out_x, out_hidden
    


    
class lambda_LSTM(nn.Module):
    def __init__(self,len_lambda,arg_nn) -> None:
        super(lambda_LSTM, self).__init__()
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net_lambda = nn.LSTM(input_size=len_lambda, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size, len_lambda)
        
    def forward(self, grad_lambda, h_s = None):
        # Reshape lambda to match the input size of the LSTM
        grad_lambda = grad_lambda.view(-1, self.len_lambda)
        if h_s is None: 
            out_temp = self.net_lambda(grad_lambda)
        else:
            out_temp = self.net_lambda(grad_lambda, h_s)
        out_lambda = out_temp[0]
        out_hidden = out_temp[1]
        delta_lambda = self.net_fc(out_lambda)
        
        return delta_lambda, out_hidden
    

class Discriminator(nn.Module):
    def __init__(self, len_x, len_lambda, arg_nn) -> None:
        super(Discriminator, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net = nn.Sequential(
            nn.Linear(len_x + len_lambda, arg_nn.hidden_size_x),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size_x, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, r):
        x = x.view(-1, self.len_x)
        r = r.view(-1, self.len_lambda)
        xr = torch.cat((x, r), 1)
        out = self.net(xr)
        return out
    
def lambda_proj(r):
    alpha = 2
    _r = torch.where(r < -1, -alpha * r - (alpha - 1), r)
    _r = torch.where(r > 1, alpha * r - (alpha - 1), _r)
    _r = torch.where((r >= -1) & (r <= 1), r ** alpha, _r)
    return _r

def x_proj(x):
    pass

if __name__ == "__main__":
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 8
    len_x = 3
    len_lambda = 2 * len_x +1
    batch_size = 1

    x_model = x_LSTM(len_x, arg_nn)
    
    lambda_model = lambda_LSTM(len_lambda, arg_nn)

    r = torch.randn(batch_size, len_lambda)
    x = torch.randn(batch_size, len_x)

    x_model.train()

    lambda_model.train()

    #r.requires_grad = True

 
    delta_x = x_model(x)
    delta_lambda = lambda_model(r)

    pass
    

 