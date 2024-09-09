import torch
import numpy as np
from nn import *
from sys_prob import *
from collections import namedtuple,deque
from random import sample
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def my_train(prob, x_model, L_model,r, num_epochs, num_steps,optimizer):
 
    total_loss = 0 # to be determined, loss should include the loss of x and L, and also lambda
    (x_optimizer, L_optimizer, r_optimizer) = optimizer
    #x_model.train()
    #L_model.train()
    #data_set = {"x":[],"r":[],"L":[]}
    
    # initialize a projection of lambda
    #r = torch.randn(5, len_lambda)
     #_r is the projected r

    # the loss for L update
    loss_MSE = torch.nn.MSELoss()
    
    class data:
        def __init__(self,size=5000):
            self.lambda_past = []
            self.x_past = []
            self.L_past = []
            self.size = size
        def reset(self):
            self.lambda_past = []
            self.x_past = []
            self.L_past = []
        def append(self, x, r, L):
            self.lambda_past.append(r.detach().numpy())
            self.x_past.append(x.detach().numpy())
            self.L_past.append(L)
            if len(self.lambda_past)>self.size:
                self.lambda_past.pop(0)
                self.x_past.pop(0)
                self.L_past.pop(0)
    data_epoch = data()
    writer.add_graph(x_model, r_proj(r))
    writer.add_graph(L_model, [x_model(r_proj(r)),r_proj(r)])
    writer.close()
    # freeze x model and learn L model
    for epoch in range(num_epochs):

        for step in range(num_steps):
            # in steps, we update x by derive new data. However, this process is better put before L update
            # here we also want to generate some data for L update
            for _ in range(5):
                for param in x_model.parameters():
                    param.requires_grad = True
                for param in L_model.parameters():
                    param.requires_grad = False
                r.requires_grad = False
            
            
                x_model.zero_grad()
                _r = r_proj(r)
                x_ = x_model(_r)
                L_ = torch.sum(L_model(x_,_r))
                L_truth = torch.tensor(prob(x_.detach().numpy(),_r.detach().numpy()),dtype=torch.float32)
                data_epoch.append(x_,_r,L_truth)
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
            # obtaing L_truth here is not necessary
                r_optimizer.zero_grad()
                _r = r_proj(r)
                L_out = -torch.sum(L_model(x_model(_r),_r))
                L_out.backward()
                r_optimizer.step()
            
        for param in x_model.parameters():
            param.requires_grad = False
        for param in L_model.parameters():
            param.requires_grad = True
            #param.requires_grad = False        
        r.requires_grad = False
        # return the ground truth of L given x and r, not defined yet
        # here we may need a new class inherit from prob
        
        #_r = r_proj(r)
        # here is the only function that needs to be defined
        #L_truth = torch.tensor(prob(x_model(_r).detach().numpy(),_r.detach().numpy()),dtype=torch.float32)
        for _ in range(30):
            L_optimizer.zero_grad()

            id_sample = sample(range(len(data_epoch.x_past)),min(50,len(data_epoch.x_past)))
            
            x_data = torch.tensor(data_epoch.x_past,dtype=torch.float32)[id_sample]
            r_data = torch.tensor(data_epoch.lambda_past,dtype=torch.float32)[id_sample]
            L_truth = torch.tensor(data_epoch.L_past,dtype=torch.float32)[id_sample]
            #loss_L = loss_MSE(L_model(x_model(_r),_r),L_truth)
            #loss_L = loss_CE(L_model(x_model(_r),_r),L_truth)
            loss_L = loss_MSE(L_model(x_data.view(-1,len_x),r_data.view(-1,len_lambda)),L_truth.view(-1,1))
            loss_L.backward()
            L_optimizer.step()
        

        print("L loss", loss_L.detach().numpy(),'delta',out-x_[0].detach().numpy())

        # here we update L model
    r_final = r_proj(r)    
    print("lambda:",r_final)
    print("x:",x_model(r_final).detach().numpy())


            





if __name__ == "__main__":

    np.random.seed(10000)
    L = problem_generator()
    out = L.solve().x
    #print(out)
    #what
    
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 16
    len_x = 5
    len_lambda = 2 * len_x +1
    num_epochs = 3000
    num_steps = 5

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_LSTM(len_x, len_lambda, arg_nn)
    x_optimizer = torch.optim.SGD(x_model.parameters(), lr=0.001)
    L_optimizer = torch.optim.SGD(L_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda) # the first input is the batch size, r itself is parameter to be learned
    r_optimizer = torch.optim.SGD([r], lr=0.001)
    optimizer = (x_optimizer, L_optimizer, r_optimizer)

    # in this setting, L and Batch are the same. we don't consider a batch
    
    x = x_model(r)
    loss = L_model(x,r)

    L_truth = L(x.detach().numpy(),r.detach().numpy())
    pass

    
    my_train(L, x_model, L_model,r, num_epochs, num_steps, optimizer)

    print(out)
