import torch
import numpy as np
from nn import *
from sys_prob import *


def my_train(prob, x_model, L_model,r, num_epochs, num_steps,optimizer):
 
    total_loss = 0 # to be determined, loss should include the loss of x and L, and also lambda
    (x_optimizer, L_optimizer, r_optimizer) = optimizer
    x_model.train()
    L_model.train()
    
    # initialize a projection of lambda
    #r = torch.randn(5, len_lambda)
     #_r is the projected r

    # the loss for L update
    loss_MSE = torch.nn.MSELoss()
    class data:
        lambda_past = []
        x_past = []
        L_past = []
        def reset(self):
            self.lambda_past = []
            self.x_past = []
            self.L_past = []
    data_epoch = data()
    
    # freeze x model and learn L model
    for epoch in range(num_epochs):

        for step in range(num_steps):
            # in steps, we update x by derive new data. However, this process is better put before L update
            # here we also want to generate some data for L update
            for param in x_model.parameters():
                param.requires_grad = True
            for param in L_model.parameters():
                param.requires_grad = False
            r.requires_grad = False
            

            x_model.zero_grad()
            _r = r_proj(r)
            x_ = x_model(_r)
            L_ = L_model(x_,_r)
            L_.backward()
            x_optimizer.step()
            # store one piece of data
            #data_epoch.x_past.append(x_.detach().numpy())
            #data_epoch.lambda_past.append(r.detach().numpy()) # store original r
            #data_epoch.L_past.append(L_.detach().numpy()) # this should be converted to the ground truth

            for param in x_model.parameters():
                param.requires_grad = False
            for param in L_model.parameters():
                param.requires_grad = False
            r.requires_grad = True

            # r update
            r_optimizer.zero_grad()
            _r = r_proj(r)
            L_out = L_model(x_model(_r),_r)
            -L_out.backward()
            r_optimizer.step()
            
        for param in x_model.parameters():
            param.requires_grad = False
        for param in L_model.parameters():
            param.requires_grad = True    
        r.requires_grad = False
        # return the ground truth of L given x and r, not defined yet
        # here we may need a new class inherit from prob
        
        _r = r_proj(r)
        # here is the only function that needs to be defined
        L_truth = prob(x_model(_r).detach().numpy(),_r)
        
        L_optimizer.zero_grad()
        loss_L = loss_MSE(L_model(x_model(_r),_r),L_truth)
        loss_L.backward()
        L_optimizer.step()

        # here we update L model


            





if __name__ == "__main__":
    
    L = problem_generator()
    
    
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 8
    len_x = 5
    len_lambda = 2 * len_x +1
    num_epochs = 100
    num_steps = 10

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_LSTM(len_x, len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    L_optimizer = torch.optim.Adam(L_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda) # the first input is the batch size, r itself is parameter to be learned
    r_optimizer = torch.optim.Adam([r], lr=0.001)
    optimizer = (x_optimizer, L_optimizer, r_optimizer)

    # in this setting, L and Batch are the same. we don't consider a batch
    
    x = x_model(r)
    loss = L_model(x,r)

    L_truth = L(x.detach().numpy(),r.detach().numpy())
    pass

    
    my_train(L, x_model, L_model,r, 100, 10, optimizer)
    pass