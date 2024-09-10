import torch
import torch.nn as nn

# this is the centralized version without a projection layer 
class x_LSTM(nn.Module):
    # feed lambda and output x. Since centralized, we do not consider global consensus terms 
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(x_LSTM, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net_x = nn.LSTM(input_size=len_lambda, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size, len_x)

    # r represents lambda
    def forward(self, grad_r):
        # Reshape x and lambda to match the input size of the LSTM
        grad_r = grad_r.view(-1, self.len_lambda)
        # Pass x through the LSTM
        # the output is a tuple, we only need the first element
        out_temp = self.net_x(grad_r)
        out_lambda = out_temp[0]
        out_hidden = out_temp[1]
        # Pass lambda through the LSTM
        delta_x = self.net_fc(out_lambda)
        return delta_x

class L_MLP(nn.Module):
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(L_MLP, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        # let the nn to learn the representation of a obj func and len-lambda constraint functions
        self.net_fg = nn.Sequential(
            nn.Linear(len_x, arg_nn.hidden_size_x),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size_x, arg_nn.hidden_size_x),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size_x, len_lambda+1)
        )
        self.net_L_o = nn.Linear(2 * len_lambda + 1, 1)


    def forward(self, x, r):
        # Reshape x and lambda to match the input size of the LSTM
        x = x.view(-1, self.len_x)
        out_fg = self.net_fg(x)
        temp = torch.cat((out_fg,r), dim=1)
        out = self.net_L_o(temp)
        return out
    
class lambda_LSTM(nn.Module):
    def __init__(self,len_lambda,arg_nn) -> None:
        super(lambda_LSTM, self).__init__()
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net_lambda = nn.LSTM(input_size=len_lambda, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size, len_lambda)
        
    def forward(self, grad_lambda):
        # Reshape lambda to match the input size of the LSTM
        grad_lambda = grad_lambda.view(-1, self.len_lambda)
        out_temp = self.net_lambda(grad_lambda)
        out_lambda = out_temp[0]
        out_hidden = out_temp[1]
        delta_lambda = self.net_fc(out_lambda)
        return delta_lambda
    
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
    batch_size = 5

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_MLP(len_x, len_lambda, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)

    r = torch.randn(batch_size, len_lambda)
    x = torch.randn(batch_size, len_x)

    x_model.train()
    L_model.train()
    lambda_model.train()

    #r.requires_grad = True

    _L = L_model(x,r)
    delta_x = x_model(r)
    delta_lambda = lambda_model(r)

    pass
    

 