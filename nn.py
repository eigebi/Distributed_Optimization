import torch
import torch.nn as nn
import torch.nn.functional as F



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
            out1, _ = self.lstm1(out, (h02, c02))
            
            h02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
            c02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
            out2, _ = self.lstm2(out1, (h02, c02))
            
            out = self.fc(out2[:, -1, :])
            return out
        

        


class x_LSTM(nn.Module):
    def __init__(self,len_x,len_lambda,arg_nn) -> None:
        super(x_LSTM, self).__init__()
        self.len_x = len_x
        self.len_lambda = len_lambda
        self.arg_nn = arg_nn
        self.net_x = nn.LSTM(input_size=len_lambda, hidden_size=arg_nn.hidden_size)
        self.net_fc = nn.Linear(arg_nn.hidden_size,len_x)


    # r represents lambda
    def forward(self, r):
        # Reshape x and lambda to match the input size of the LSTM
        r = r.view(-1, self.len_lambda)

        
        # Pass x through the LSTM
        out_r = self.net_x(r)
        
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
        self.net_L = nn.LSTM(input_size=len_lambda, hidden_size=arg_nn.hidden_size)

    def forward(self, x, r):
        # Reshape x and lambda to match the input size of the LSTM
        x = x.view(-1, self.len_x)
        
        out_f = self.net_f(x)
        # concatenate x and out_f, make linear connection to them, with all positive weights
        

        # concatenate x and out_f
        concatenated = torch.cat((x, out_f), dim=1)

        # make linear connection with all positive weights
        linear = F.relu(concatenated)

        # Pass the linear connection through the LSTM
        out_x, _ = self.net_L(linear)

        return out_x, out_r
        
        return out_x, out_r
    









def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Define the hyperparameters
    input_size = 28
    hidden_size = 128
    num_layers = 2
    output_size = 10
    num_epochs = 5
    learning_rate = 0.001
    
    # Create the model
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Save the model
    torch.save(model.state_dict(), "lstm_model.pth")
    
    # Load the model
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load("lstm_model.pth"))
    
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Accuracy: {100 * correct / total}%")