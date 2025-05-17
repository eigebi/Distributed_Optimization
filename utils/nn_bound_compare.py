import torch
import torch.nn as nn

# this is the centralized version without a projection layer 
class x_LSTM(nn.Module):
    # feed lambda and output x. Since centralized, we do not consider global consensus terms 
    def __init__(self,len_x,len_lambda, len_feature, arg_nn, bounded=False) -> None:
        super(x_LSTM, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.len_feature = len_feature
        self.arg_nn = arg_nn
        self.bounded = bounded
        # to feed information of primal problem, we take grad_x as input
        self.net_x = nn.LSTM(input_size=len_lambda+len_feature, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size, len_x)
        self.net_tanh = nn.Tanh()

    # r represents lambda
    def forward(self, r, feature, h_s = None):
        # Reshape x and lambda to match the input size of the LSTM
        r = r.view(-1, self.len_lambda)
        feature = feature.view(-1, self.len_feature)
        # concatenate x and lambda
        r_f = torch.cat((r,feature),dim=1)
        # Pass x through the LSTM
        # the output is a tuple, we only need the first element
        if h_s is None:
            out_temp = self.net_x(r_f)
        else:
            out_temp = self.net_x(r_f, h_s)
        out_lambda = out_temp[0]
        out_hidden = out_temp[1]
        # Pass lambda through the LSTM
        out_x = self.net_fc(out_lambda)
        #out_x = 15*torch.tanh(out_x)
        if self.bounded:
            # map from [-1,1] to [lb,ub]
            u_b = torch.tensor(self.arg_nn.u_b[:-1],dtype=torch.float32)
            l_b = 0
            out_x = (self.net_tanh(out_x) + 1) * (u_b-l_b)/2 + l_b
        return out_x


class x_MLP(nn.Module):
    # feed lambda and output x. Since centralized, we do not consider global consensus terms 
    def __init__(self,len_x,len_lambda, len_feature, arg_nn, bounded=False) -> None:
        super(x_MLP, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.len_feature = len_feature
        self.arg_nn = arg_nn
        self.bounded = bounded
        # to feed information of primal problem, we take grad_x as input
        self.net_feature = nn.Sequential(
            nn.Linear(len_feature, arg_nn.hidden_size),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size, arg_nn.hidden_size),
            nn.ReLU())
            
        self.net_fc = nn.Sequential(
            nn.Linear(arg_nn.hidden_size+len_lambda, arg_nn.hidden_size),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size, arg_nn.hidden_size),
            nn.ReLU(),
            nn.Linear(arg_nn.hidden_size, len_x))
        self.net_tanh = nn.Tanh()
        

    # r represents lambda
    def forward(self, r, feature):
        # Reshape x and lambda to match the input size of the LSTM
        r = r.view(-1, self.len_lambda)
        feature = feature.view(-1, self.len_feature)

        temp_feature = self.net_feature(feature)


        # concatenate x and lambda
        r_f = torch.cat((r,temp_feature),dim=1)
        # Pass x through the LSTM
        # the output is a tuple, we only need the first element
        out_x = self.net_fc(r_f)
        
        if self.bounded:
            # map from [-1,1] to [lb,ub]
            u_b = torch.tensor(self.arg_nn.u_b[:-1],dtype=torch.float32)
            l_b = 0
            out_x = (self.net_tanh(out_x) + 1) * (u_b-l_b)/2 + l_b
        return out_x


class L_MLP(nn.Module):
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(L_MLP, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        # let the nn to learn the representation of a obj func and len-lambda constraint functions
        self.net_fg = nn.Sequential(
            nn.Linear(len_x, arg_nn.hidden_size_x),
            nn.Tanh(),
            nn.Linear(arg_nn.hidden_size_x, arg_nn.hidden_size_x),
            nn.Tanh(),
            nn.Linear(arg_nn.hidden_size_x, len_lambda+1)
        )
        self.net_L_o = nn.Linear(2 * len_lambda + 1, 1)

        # Initialize the weights of the linear layers
        nn.init.xavier_uniform_(self.net_fg[0].weight)
        nn.init.xavier_uniform_(self.net_fg[2].weight)
        nn.init.xavier_uniform_(self.net_fg[4].weight)
        nn.init.xavier_uniform_(self.net_L_o.weight)

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
        #delta_lambda = 1*torch.tanh(delta_lambda)
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
    

 