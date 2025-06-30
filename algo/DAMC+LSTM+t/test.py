import numpy as np
import torch
from nn_CLSTM import x_LSTM

ub = 3

def lambda_proj(r):
    alpha = 2
    _r = np.where(r < -1, -alpha * r - (alpha - 1), r)
    _r = np.where(r > 1, alpha * r - (alpha - 1), _r)
    _r = np.where((r >= -1) & (r <= 1), r ** alpha, _r)
    # using softplus instead of 
    #return np.log(1+np.exp(r)) 
    return _r

def derive_grad_lambda(z, r):
    # x, z: (num_problem, num_var)
    alpha = 2
    dr = np.where(r < -1, -alpha, 0)
    dr = np.where(r > 1, alpha, dr)
    dr = np.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    dr = dr.astype(np.float32)
    #dr = 1-1/(1+np.exp(r))
    partial_grad_lambda_p = np.concatenate([-z,z-6,np.sum(z, keepdims=True)-ub], axis=0)
    grad_lambda = dr * partial_grad_lambda_p 
    return grad_lambda
def derive_grad_lambda_with_x(x, r):
    # x, z: (num_problem, num_var)
    x1 = x[0][0]
    x2 = x[1][:,0]
    alpha = 2
    dr = np.where(r < -1, -alpha, 0)
    dr = np.where(r > 1, alpha, dr)
    dr = np.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    dr = dr.astype(np.float32)
    #dr = 1-1/(1+np.exp(r))
    partial_grad_lambda_p = np.array([-x1[0],-x2[1],x1[0]-6,x2[1]-6,x2[0]+x2[1]-ub],dtype=np.float32)
    grad_lambda = dr * partial_grad_lambda_p 
    return grad_lambda

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    obj = [lambda x: (x+2)**2, lambda x: (x-5)**2]
    jac = [lambda x: 2*(x+2), lambda x: 2*(x-5)]
    obj = [lambda x: 1/(1+np.exp(-2*(x-3))), lambda x: 1/(1+np.exp(-1*(x-10)))]
    jac = [lambda x: -obj[0](x)*(1-obj[0](x)), lambda x: -obj[1](x)*(1-obj[1](x))]
    #jac = [lambda x:-1/(x+1)/np.log(1+1),lambda x:-1/(x+1)/np.log(5+1)]
    # add constraints x1+x2 < ub, this constraint is assigned to agent 1
    #r = np.zeros(5, dtype=np.float32)  # Initialize r (lambda)
    #z = np.array([0.5, 0.5], dtype=np.float32)  # Example z(k) values
    len_x = [1,2]
    len_lambda = 5  # Number of lambda variables
    #gamma = [np.zeros([i], dtype=np.float32) for i in len_x]  # Initialize gamma
    rho = 50
    beta = 40

    

    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    
    con_assignment = np.array([1])  # Assign constraints to agents
    x_models = [x_LSTM(1,arg_nn) for _ in range(2)]  # Create two LSTM models for x1 and x2
    x_optimizers = [torch.optim.Adam(x_models[i].parameters(), lr=0.001) for i in range(2)]  # Create optimizers for each model
    #z = lambda x: np.exp(-x) + 2 + 1 * np.sin(x)  # Example function for z(k)
    lambda_model = x_LSTM(1, arg_nn)  # Create LSTM model for lambda
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)
    #reserved_r = r
    LSTM_lambda = True  # Use LSTM for lambda updates



    #x = [torch.randn(1,1), torch.randn(1,2)]  # Initialize x for each model
    num_layer = 2
    num_problem = 1
    num_epoch = 50
    num_iter = 500
    num_frame = 30
    zt = 10 # update frequency for z and gamma

    for k in range(num_epoch):
        r = np.zeros(5, dtype=np.float32)  # Initialize r (lambda)
        reserved_r = r
        x = [np.random.randn(1,1).astype(np.float32), np.random.randn(2,1).astype(np.float32)] # initialize x for each model
        reserved_x = x
        z = np.array([-2, -0.5], dtype=np.float32)  # Example z(k) values
        gamma = [np.zeros([i], dtype=np.float32) for i in len_x]  # Initialize gamma
        

        hidden_x = [(torch.randn(num_layer, len_x[i], arg_nn.hidden_size), torch.randn(num_layer, len_x[i], arg_nn.hidden_size)) for i in range(2)]
        hidden_lambda = (torch.randn(num_layer, len_lambda, arg_nn.hidden_size), torch.randn(num_layer, len_lambda, arg_nn.hidden_size)) # (num_layer, batch_size, hidden_size)


        for f in range(num_iter):
            for j in range(num_frame):
                # update x
                # pay attention that here lambda should be projected or clipped
                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0] - z[0]))
                delta, hidden_temp = x_models[0](torch.tensor(grad_x), hidden_x[0])
                hidden_x[0] = (hidden_temp[0].detach(), hidden_temp[1].detach())
                _x = torch.tensor(x[0]) + delta[0]
                x[0] = _x.detach().numpy()
                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0] - z[0]))
                _x.backward(torch.tensor(grad_x), retain_graph=True)

                

                grad_x = np.zeros_like(x[1])
                grad_x[1,:] = jac[1](x[1][1,:]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][1,:] - z[1]))
                grad_x[0,:] = r[4] + gamma[1][0] + rho * (x[1][0,:] - z[0]) 
                delta, hidden_temp = x_models[1](torch.tensor(grad_x), hidden_x[1])
                hidden_x[1] = (hidden_temp[0].detach(), hidden_temp[1].detach())
                _x = torch.tensor(x[1]) + delta[0]
                x[1] = _x.detach().numpy()
                grad_x = np.zeros_like(x[1])
                grad_x[1,:] = jac[1](x[1][1,:]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][1,:] - z[1]))
                grad_x[0,:] = r[4] + gamma[1][0] + rho * (x[1][0,:] - z[0])


                #loss = -obj[0](_x[:,0]) - obj[1](_x[:,1]) + torch.tensor(gamma[1])@(_x[:,1]-torch.tensor(z)) + rho/2*torch.norm(_x[:,1]-torch.tensor(z))**2
                _x.backward(torch.tensor(grad_x), retain_graph=True)
                #loss.backward(retain_graph=True)

                



                # update z and gamma
                if j % zt == 0:
                    gamma[0] = gamma[0] + rho * (x[0] - z[0])[0]
                    gamma[1] = gamma[1] + rho * (x[1] - z[:,np.newaxis])[:,0]
                    z[0] = (gamma[0] + rho * x[0] + gamma[1][0] + rho * x[1][0,:]+ beta * z[0])[0,0] / (2 * rho + beta)
                    z[1] = (gamma[1][1] + rho * x[1][1,:] + beta * z[1])[0] / (rho + beta)

                # update lambda
                if j % zt == 0:
                    if not LSTM_lambda:
                        # update lambda using gradient ascent
                        r = np.maximum(r + 0.05 * (np.concatenate([-z,z-6,np.sum(z, keepdims=True)-ub], axis=0)), 0)
                    else:
                        # alternatively, using LSTM; here lambda is unprojected
                        r = reserved_r
                        grad_lambda = derive_grad_lambda(z, r)
                        #grad_lambda = derive_grad_lambda_with_x(x,r)
                        delta_r, hidden_lambda_temp = lambda_model(torch.tensor(grad_lambda[:,np.newaxis]), hidden_lambda)
                        hidden_lambda = (hidden_lambda_temp[0].detach(), hidden_lambda_temp[1].detach())
                        _r = torch.tensor(r) + 0.5 * delta_r[0,:,0]
                        r = _r.detach().numpy()
                        grad_lambda = derive_grad_lambda(z, r)
                        #grad_lambda = derive_grad_lambda_with_x(x,r)
                        _r.unsqueeze(1).backward(-torch.tensor(grad_lambda[:,np.newaxis]), retain_graph=True)
                        reserved_r = r
                        r = lambda_proj(r)









            for i in range(2):
                x_optimizers[i].step()
                x_optimizers[i].zero_grad()
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            
            print("x:",x,"lambda:",r)
            print("z:", z," gamma:", gamma)
            #print("obj: ", obj[0](x[0].numpy())+obj[1](x[1][:,1].numpy()))







    for k in range(6):
        r = np.zeros(5, dtype=np.float32)  # Initialize r (lambda)
        reserved_r = r
        z = np.array([0, 0], dtype=np.float32)  # Example z(k) values
        gamma = [np.zeros([i], dtype=np.float32) for i in len_x]  # Initialize gamma
        x = [torch.randn(1,1), torch.randn(1,2)]
        LSTM_lambda = True
        for n in range(2):
            for param in x_models[n].parameters():
                param.requires_grad = False
        for f in range(500):
            for j in range(10):
                if not LSTM_lambda:
                    # update lambda using gradient ascent
                
                    r = np.maximum(r + 0.5 * (np.concatenate([-z,z-6,np.sum(z, keepdims=True)-1], axis=0)), 0)
                else:
                # alternatively, using LSTM
                    r = reserved_r
                    #grad_lambda = derive_grad_lambda(z, r)
                    grad_lambda = derive_grad_lambda_with_x(x,r)
                    delta_r, _ = lambda_model(torch.tensor(grad_lambda[np.newaxis,:], dtype=torch.float32))
                    _r = torch.tensor(r,dtype=torch.float32) + delta_r[0,0]
                    r = _r.detach().numpy()
                    #grad_lambda = derive_grad_lambda(z, r)
                    grad_lambda = derive_grad_lambda_with_x(x,r)
                    _r.unsqueeze(0).backward(-torch.tensor(grad_lambda[np.newaxis,:],dtype=torch.float32), retain_graph=True)
                    reserved_r = r
                    r = lambda_proj(r)
                

                
                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0].numpy() - z[0]))
                delta, _ = x_models[0](grad_x)
                _x = x[0] + delta[0]
                x[0] = _x.detach()

                #x[0] = torch.clip(x[0], -10, 10)  # Clip x[0] to a range, if necessary

                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0].numpy() - z[0]))
                #_x.backward(grad_x, retain_graph=True)
                gamma[0] = gamma[0] + rho * (x[0].numpy() - z[0])[0]

                grad_x = torch.zeros_like(x[1])
                grad_x[:,1] = jac[1](x[1][:,1]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][:,1].numpy() - z[1]))
                grad_x[:,0] = torch.tensor(r[4] + gamma[1][0] + rho * (x[1][:,0].numpy() - z[0]))
                delta, _ = x_models[1](grad_x)
                _x = x[1] + delta[0]

                x[1] = _x.detach()

                #x[1] = torch.clip(x[1], -10, 10)  # Clip x[1] to a range, if necessary

                grad_x = torch.zeros_like(x[1])
                grad_x[:,1] = jac[1](x[1][:,1]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][:,1].numpy() - z[1]))
                grad_x[:,0] = torch.tensor(r[4] + gamma[1][0] + rho * (x[1][:,0].numpy() - z[0]))
                #_x.backward(grad_x, retain_graph=True)
                gamma[1] = gamma[1] + rho * (x[1].numpy() - z)[0]

                z[0] = (gamma[0] + rho * x[0].numpy() + gamma[1][0] + rho * x[1][:,0].numpy()+ beta * z[0])[0,0] / (2 * rho + beta)
                z[1] = (gamma[1][1] + rho * x[1][:,1].numpy() + beta * z[1])[0] / (rho + beta)
                #z = np.ones(2)
                pass

            #for i in range(2):
                #x_optimizers[i].step()
                #x_optimizers[i].zero_grad()
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            #print("x:",x,"lambda:",r)
            #print("z:", z," gamma:", gamma)
            print("obj: ", obj[0](x[0].numpy())+obj[1](x[1][:,1].numpy()))
    for k in range(10):
        r = np.zeros(5, dtype=np.float32)  # Initialize r (lambda)
        z = np.array([-1, 5], dtype=np.float32)  # Example z(k) values
        gamma = [np.zeros([i], dtype=np.float32) for i in len_x]  # Initialize gamma
        x = [10*torch.randn(1,1), torch.randn(1,2)]
        reserved_r = r
        LSTM_lambda = True
        for n in range(2):
            for param in x_models[n].parameters():
                param.requires_grad = True
        for f in range(500):
            for j in range(10):
                if not LSTM_lambda:
                    # update lambda using gradient ascent
                
                    r = np.maximum(r + 0.1 * (np.concatenate([-z,z-6,np.sum(z, keepdims=True)-1], axis=0)), 0)
                else:
                # alternatively, using LSTM
                    r = reserved_r
                    grad_lambda = derive_grad_lambda(z, r)
                    delta_r, _ = lambda_model(torch.tensor(grad_lambda[np.newaxis,:], dtype=torch.float32))
                    _r = torch.tensor(r,dtype=torch.float32) + delta_r[0,0]
                    r = _r.detach().numpy()
                    grad_lambda = derive_grad_lambda(z, r)
                    #grad_lambda = derive_grad_lambda_with_x(x,r)
                    _r.unsqueeze(0).backward(-torch.tensor(grad_lambda[np.newaxis,:],dtype=torch.float32), retain_graph=True)
                    reserved_r = r
                    r = lambda_proj(r)
                

                
                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0].numpy() - z[0]))
                delta, _ = x_models[0](grad_x)
                _x = x[0] + delta[0]
                x[0] = _x.detach()

                #x[0] = torch.clip(x[0], -10, 10)  # Clip x[0] to a range, if necessary

                grad_x = jac[0](x[0]) + (-r[0]+r[2] + gamma[0] + rho * (x[0].numpy() - z[0]))
                _x.backward(grad_x, retain_graph=True)
                gamma[0] = gamma[0] + rho * (x[0].numpy() - z[0])[0]

                grad_x = torch.zeros_like(x[1])
                grad_x[:,1] = jac[1](x[1][:,1]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][:,1].numpy() - z[1]))
                grad_x[:,0] = torch.tensor(r[4] + gamma[1][0] + rho * (x[1][:,0].numpy() - z[0]))
                delta, _ = x_models[1](grad_x)
                _x = x[1] + delta[0]

                x[1] = _x.detach()

                #x[1] = torch.clip(x[1], -10, 10)  # Clip x[1] to a range, if necessary

                grad_x = torch.zeros_like(x[1])
                grad_x[:,1] = jac[1](x[1][:,1]) + (-r[1] + r[3]+ r[4] + gamma[1][1] + rho * (x[1][:,1].numpy() - z[1]))
                grad_x[:,0] = torch.tensor(r[4] + gamma[1][0] + rho * (x[1][:,0].numpy() - z[0]))
                _x.backward(grad_x, retain_graph=True)
                gamma[1] = gamma[1] + rho * (x[1].numpy() - z)[0]

                z[0] = (gamma[0] + rho * x[0].numpy() + gamma[1][0] + rho * x[1][:,0].numpy()+ beta * z[0])[0,0] / (2 * rho + beta)
                z[1] = (gamma[1][1] + rho * x[1][:,1].numpy() + beta * z[1])[0] / (rho + beta)
                #z = np.ones(2)
                pass

            for i in range(2):
                x_optimizers[i].step()
                x_optimizers[i].zero_grad()
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            #print("x:",x,"lambda:",r)
            #print("z:", z," gamma:", gamma)
            print("obj: ", obj[0](x[0].numpy())+obj[1](x[1][:,1].numpy()))
            print("lambda: ",r)
