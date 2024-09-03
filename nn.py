import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter


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
        out_hidden = out_r[1]
        
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

def r_proj(r,alpha):
    r[r<-1] = -alpha*r[r<-1]-(alpha-1)
    r[r>1] = alpha*r[r>1]-(alpha-1)
    r[(r>=-1) & (r<=1)] = r[(r>=-1) & (r<=1)]**alpha



def my_train(prob, x_model, L_model, num_epochs, num_steps):
 
    total_loss = 0

    r = torch.abs(torch.randn(5, len_lambda))
    r = r_proj(r,0.5)
    #make r to be positive
    for epoch in range(num_epochs):
        x_model.eval()
        L_model.train()
        # return the ground truth of L given x and r
        L_truth = prob(x_model(r),r)
        loss_MSE = torch.nn.MSELoss()
        L_optimizer.zero_grad()
        loss_L = loss_MSE(L_model(x_model(r),r),L_truth)
        loss_L.backward()
        L_optimizer.step()



        for step in range(num_steps):
            x_model.train()
            L_model.eval()
            lambda_past = []

            if step % 10 == 0: # this if may not be necessary
                x_optimizer.zero_grad()
                L_out = L_model(x_model(lambda_past),lambda_past)
                loss_x = torch.sum(L_out)
                loss_x.backward()
                x_optimizer.step()
                lambda_past = []
        x_model.eval()
        L_model.eval()
        r.requires_grad = True
        r_optimizer.zero_grad()
        L_out = L_model(x_model(r),r)
        -L_out.backward()
        r_optimizer.step()

        






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

    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    L_optimizer = torch.optim.Adam(L_model.parameters(), lr=0.001)
    r_optimizer = torch.optim.Adam([r], lr=0.001)
    # test passed 
    # construct learning process








'''
if __name__ == "__main__":
    # Define the hyperparameters
    input_size = 28
    hidden_size = 128
    num_layers = 2
    output_size = 10
    num_epochs = 5
    learning_rate = 0.001
    len_x = 3
    len_lambda = 2 * len_x +1
    
    # Create the model
    model_x = x_LSTM(len_x, len_lambda, arg_nn)
    model_L = L_LSTM(input_size, hidden_size, num_layers, output_size)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    opt_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate)
    opt_L = torch.optim.Adam(model_L.parameters(), lr=learning_rate)


    
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
    '''