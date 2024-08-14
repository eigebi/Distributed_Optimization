import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    class DualLSTM(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, num_layers1, num_layers2, output_size):
            super(nn.Module, self).__init__()
            self.hidden_size1 = hidden_size1
            self.hidden_size2 = hidden_size2
            self.num_layers1 = num_layers1
            self.num_layers2 = num_layers2
            self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers1, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers2, batch_first=True)
            self.fc = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            h01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)
            c01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)
            out1, _ = self.lstm1(x, (h01, c01))
            
            h02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
            c02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
            out2, _ = self.lstm2(out1, (h02, c02))
            
            out = self.fc(out2[:, -1, :])
            return out
        


class CLSTM(nn.Module):
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(CLSTM, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn

    def

