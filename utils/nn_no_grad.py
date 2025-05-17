import torch
import torch.nn as nn
import torch.nn.functional as F



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
    def forward(self, r):
        # Reshape x and lambda to match the input size of the LSTM
        r = r.view(-1, self.len_lambda)

        
        # Pass x through the LSTM
        # the output is a tuple, we only need the first element
        out_temp = self.net_x(r)
        out_r = out_temp[0]
        out_hidden = out_temp[1]
        
        # Pass lambda through the LSTM
        out_x = self.net_fc(out_r)
        
        return out_x

class L_LSTM(nn.Module):
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(L_LSTM, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net_f = nn.LSTM(input_size=len_x, hidden_size=arg_nn.hidden_size)
        self.net_fc_i = nn.Linear(arg_nn.hidden_size, arg_nn.hidden_size_x)
        self.net_ReLU = nn.ReLU()
        self.net_fc_o = nn.Linear(arg_nn.hidden_size_x + len_lambda, 1)


    def forward(self, x, r):
        # Reshape x and lambda to match the input size of the LSTM
        x = x.view(-1, self.len_x)
        
        out_temp = self.net_f(x)
        out_f = out_temp[0]
        out_hidden = out_temp[1]
        f_g = self.net_ReLU(self.net_fc_i(out_f))

        temp = torch.cat((f_g,r), dim=1)

        # make linear connection with all positive weights
        out = self.net_fc_o(temp)

        return out
    
def r_proj(r):
    _r = torch.zeros_like(r)
    id_0 = torch.where(r<-1)
    id_1 = torch.where(r>1)
    id_2 = torch.where((r>=-1) & (r<=1))
    alpha = 2
    if len(id_0[0])>0:
        _r[id_0] = -alpha*r[id_0]-(alpha-1)
    if len(id_1[0])>0:
        _r[id_1] = alpha*r[id_1]-(alpha-1)
    if len(id_2[0])>0:
        _r[id_2] = r[id_2]**alpha
    return _r



if __name__ == "__main__":
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 8
    len_x = 3
    len_lambda = 2 * len_x +1

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_LSTM(len_x, len_lambda, arg_nn)

    # in this setting, L and Batch are the same. we don't consider a batch
    r = torch.randn(5, len_lambda)
    x = x_model(r)
    loss = L_model(x,r)
    pass

 