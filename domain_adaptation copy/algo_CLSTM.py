import torch
import numpy as np
from nn_CLSTM_TL import *
from sys_prob_TL import problem_generator
#torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)
torch.random.manual_seed(10000)



def derive_grad_lambda(problems, x, r):
    alpha = 2
    dr = torch.where(r < -1, -alpha, 0)
    dr = torch.where(r > 1, alpha, dr)
    dr = torch.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    partial_grad_lambda_p = torch.tensor([problems[i].gradient_lambda(x[:,i,:].numpy()) for i in range(len(problems))],dtype=torch.float32)
    grad_lambda = dr * partial_grad_lambda_p.transpose(0,1)
    return grad_lambda

def derive_grad_x(problems, x, r):
    grad_x = torch.tensor([problems[i].gradient_x(x[:,i,:].numpy(), r[:,i,:].numpy()) for i in range(len(problems))],dtype=torch.float32)
    return grad_x.transpose(0,1)
    
    



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
    (x_optimizer, lambda_optimizer, D_optimizer_x, D_optimizer_lambda) = optimizer
    (x_model, lambda_model, D_model_x, D_model_lambda) = model
    (prob_source, prob_target) = problems
    D_loss = nn.BCELoss()
    obj_truth_result = {'source':[], 'target':[]}
    acc = []



    lambda_D_data = []
    lambda_D_label = []
    x_D_data = []
    x_D_label = []
    # to store the solution to source problems
    for n_p, prob in enumerate(prob_source):
        obj_truth_result['source'].append(prob.objective(prob.solve().x.reshape(1,-1)))
    # to store the solution to target problems
    for n_p, prob in enumerate(prob_target):
        obj_truth_result['target'].append(prob.objective(prob.solve().x.reshape(1,-1)))
    

    for epoch in range(num_epoch):

        # randomly initialize x,r and hidden states
        init_x_s = torch.randn(1, len(prob_source),len_x)
        init_r_s = torch.randn(1, len(prob_source),len_lambda)
        init_x_t = torch.randn(1, len(prob_target),len_x)
        init_r_t = torch.randn(1, len(prob_target),len_lambda)

        hidden_x_s = (torch.randn(1, len(prob_source), arg_nn.hidden_size), torch.randn(1, len(prob_source), arg_nn.hidden_size))
        hidden_lambda_s = (torch.randn(1, len(prob_source), arg_nn.hidden_size), torch.randn(1, len(prob_source), arg_nn.hidden_size))
        hidden_x_t = (torch.randn(1, len(prob_target), arg_nn.hidden_size), torch.randn(1, len(prob_target), arg_nn.hidden_size))
        hidden_lambda_t = (torch.randn(1, len(prob_target), arg_nn.hidden_size), torch.randn(1, len(prob_target), arg_nn.hidden_size))

        
       

        
        reserved_x_s = init_x_s
        reserved_r_s = init_r_s
        reserved_x_t = init_x_t
        reserved_r_t = init_r_t


        for param in x_model.parameters():
            param.requires_grad = False
        for param in lambda_model.parameters():
            param.requires_grad = False
        for param in D_model_x.parameters():
            param.requires_grad = True
        for param in D_model_lambda.parameters():
            param.requires_grad = True

        for frame in range(num_frame):
            for iter in range(num_iteration):

                # generate piecese of data for D to learn
                r_s = reserved_r_s.detach()
                x_s = reserved_x_s.detach()
                r_t = reserved_r_t.detach()
                x_t = reserved_x_t.detach()

                grad_lambda_s = derive_grad_lambda(prob_source, x_s, r_s)
                grad_lambda_t = derive_grad_lambda(prob_target, x_t, r_t)
                

                delta_lambda_s, hidden_lambda_s = lambda_model(grad_lambda_s, h_s=hidden_lambda_s)
                hidden_lambda_s = (hidden_lambda_s[0].detach(), hidden_lambda_s[1].detach())
                D_optimizer_lambda.zero_grad()
                temp = D_model_lambda(hidden_lambda_s[1].detach())
                loss_lambda = 0.1*D_loss(temp,torch.ones_like(temp))
                loss_lambda.backward(retain_graph=True)
                
                
                _r_s = r_s + delta_lambda_s
                r_s = _r_s.detach()
                r_p_s = lambda_proj(r_s)

                delta_lambda_t, hidden_lambda_t = lambda_model(grad_lambda_t, h_s=hidden_lambda_t)
                hidden_lambda_t = (hidden_lambda_t[0].detach(), hidden_lambda_t[1].detach())
                temp = D_model_lambda(hidden_lambda_t[1].detach())
                loss_lambda = D_loss(temp,torch.zeros_like(temp))
                loss_lambda.backward()
                D_optimizer_lambda.step()

                
                _r_t = r_t + delta_lambda_t
                r_t = _r_t.detach()
                r_p_t = lambda_proj(r_t)
                

                
                grad_x_s = derive_grad_x(prob_source, x_s, r_p_s)
                delta_x_s, hidden_x_s = x_model(grad_x_s, h_s=hidden_x_s)
                hidden_x_s = (hidden_x_s[0].detach(), hidden_x_s[1].detach())
                D_optimizer_x.zero_grad()
                temp = D_model_x(hidden_x_s[1].detach())
                loss_x = 0.1*D_loss(temp,torch.ones_like(temp))
                loss_x.backward(retain_graph=True)

                x_D_data.append(hidden_x_s[1].detach())
                x_D_label.append(torch.tensor(np.kron(np.ones((len(prob_target),1)),np.array([1])),dtype=torch.float32))

                _x_s = x_s + delta_x_s
                x_s = _x_s.detach()

                grad_x_t = derive_grad_x(prob_target, x_t, r_p_t)
                delta_x_t, hidden_x_t = x_model(grad_x_t, h_s=hidden_x_t)
                hidden_x_t = (hidden_x_t[0].detach(), hidden_x_t[1].detach())
                temp = D_model_x(hidden_x_t[1].detach())
                loss_x = D_loss(temp,torch.zeros_like(temp))
                loss_x.backward()
                D_optimizer_x.step()
                print("loss: ", loss_x.detach().numpy(), loss_lambda.detach().numpy())

                x_D_data.append(hidden_x_t[1].detach())
                x_D_label.append(torch.tensor(np.kron(np.ones((len(prob_target),1)),np.array([0])),dtype=torch.float32))

                _x_t = x_t + delta_x_t
                x_t = _x_t.detach()
                

                reserved_r_s = r_s
                reserved_x_s = x_s
                reserved_r_t = r_t
                reserved_x_t = x_t
        


        for param in x_model.parameters():
            param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True
        for param in D_model_x.parameters():
            param.requires_grad = False
        for param in D_model_lambda.parameters():
            param.requires_grad = False
        
        for frame in range(50):
            for iter in range(num_iteration):

                # train x and lambda
                r_s = reserved_r_s.detach()
                x_s = reserved_x_s.detach()
                r_t = reserved_r_t.detach()
                x_t = reserved_x_t.detach()

                grad_lambda_s = derive_grad_lambda(prob_source, x_s, r_s)
                grad_lambda_t = derive_grad_lambda(prob_target, x_t, r_t)
                

                delta_lambda_s, hidden_lambda_s = lambda_model(grad_lambda_s, h_s=hidden_lambda_s)
                temp = D_model_lambda(hidden_lambda_s[1])
                hidden_lambda_s = (hidden_lambda_s[0].detach(), hidden_lambda_s[1].detach())
                _r_s = r_s + delta_lambda_s
                r_s = _r_s.detach()
                grad_lambda_s = derive_grad_lambda(prob_source, x_s, r_s)
                loss_lambda = D_loss(temp,torch.ones_like(temp))
                loss_lambda.backward(retain_graph=True)
                _r_s.backward(-grad_lambda_s, retain_graph=True)
                r_p_s = lambda_proj(r_s)


                delta_lambda_t, hidden_lambda_t = lambda_model(grad_lambda_t, h_s=hidden_lambda_t)
                temp = D_model_lambda(hidden_lambda_t[1])
                hidden_lambda_t = (hidden_lambda_t[0].detach(), hidden_lambda_t[1].detach())
                _r_t = r_t + delta_lambda_t
                r_t = _r_t.detach()
                grad_lambda_t = derive_grad_lambda(prob_target, x_t, r_t)
                #混淆用标签
                loss_lambda = D_loss(temp,torch.ones_like(temp))
                loss_lambda.backward(retain_graph=True)
                _r_t.backward(-grad_lambda_t, retain_graph=True)
                r_p_t = lambda_proj(r_t)
                

                
                grad_x_s = derive_grad_x(prob_source, x_s, r_p_s)
                delta_x_s, hidden_x_s = x_model(grad_x_s, h_s=hidden_x_s)
                temp = D_model_x(hidden_x_s[1])
                hidden_x_s = (hidden_x_s[0].detach(), hidden_x_s[1].detach())
                _x_s = x_s + delta_x_s
                x_s = _x_s.detach()
                grad_x_s = derive_grad_x(prob_source, x_s, r_p_s)
                loss_x = D_loss(temp,torch.ones_like(temp))
                loss_x.backward(retain_graph=True)
                _x_s.backward(grad_x_s, retain_graph=True)


                grad_x_t = derive_grad_x(prob_target, x_t, r_p_t)
                delta_x_t, hidden_x_t = x_model(grad_x_t, h_s=hidden_x_t)
                temp = D_model_x(hidden_x_t[1])
                hidden_x_t = (hidden_x_t[0].detach(), hidden_x_t[1].detach())
                _x_t = x_t + delta_x_t
                x_t = _x_t.detach()
                grad_x_t = derive_grad_x(prob_target, x_t, r_p_t)
                loss_x = D_loss(temp,torch.ones_like(temp))
                loss_x.backward(retain_graph=True)
                _x_t.backward(grad_x_t, retain_graph=True)

                

                reserved_r_s = r_s
                reserved_x_s = x_s
                reserved_r_t = r_t
                reserved_x_t = x_t
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            x_optimizer.step()
            x_optimizer.zero_grad()
            print("loss: ", loss_x.detach().numpy(), loss_lambda.detach().numpy())
            
            # train D
            #lambda_data = torch.stack(lambda_D_data).squeeze(1).view(-1, arg_nn.hidden_size)
            #lambda_label = torch.stack(lambda_D_label).squeeze(1).view(-1, 1)
            #out_D_lambda = D_model(lambda_data)
            #loss_D_lambda = D_loss(out_D_lambda, lambda_label)
            #D_optimizer.zero_grad()
            #loss_D_lambda.backward()
            #D_optimizer.step()
            #print("loss_D_lambda: ", loss_D_lambda)





                        
            #lambda_optimizer.step()
            #lambda_optimizer.zero_grad()
            #x_optimizer.step()
            #x_optimizer.zero_grad()
            '''
            precision  = 0
            for n_p, prob in enumerate(problems):
                r_p = lambda_proj(reserved_r[n_p].view(1,-1))
                x = reserved_x[n_p].view(1,-1)
                precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result[n_p])/np.abs(obj_truth_result[n_p])
            precision /= len(problems)
            acc.append(precision)
            print("precision: ", precision)
            '''
            #print("delta: ",  [problems[i].objective(reserved_x[i].view(1,-1).detach().numpy()) for i in range(len(problems))], obj_truth_result)
            # r_p = lambda_proj(r)

            #print("L truth: ", prob(x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(x.detach().numpy()))
            #latest_r = r
            #latest_x = x
            #r = reserved_r
            #x = reserved_x
                    
        #r = latest_r
        #x = latest_x
        # Save models and accuracy array
       
        # print result each iteration
        #print("lambda:",r_p.detach().numpy())
        #print("x:",x.detach().numpy())
        #res = prob.solve()
        #print("opt x: ", res.x)
        #print("constraint function: ", prob.check_con(_x.detach().numpy())[-1])
        #print("opt obj: ", prob.objective(res.x.reshape(1,-1)))
            
    torch.save(x_model.state_dict(), 'x_model_target.pth')
    torch.save(lambda_model.state_dict(), 'lambda_model_target.pth')
    torch.save(D_model_x.state_dict(), 'D_model_x.pth')
    torch.save(D_model_x.state_dict(), 'D_model_x.pth')
    np.save('accuracy.npy', np.array(acc))

    # end of iterations
    #np.save('L_train.npy',np.array(L_train_result))
    #np.save('Loss_train.npy',np.array(Loss_train_result))
    #np.save('L_truth.npy',np.array(L_truth_result))
    #np.save('obj_train.npy',np.array(obj_truth_result))



if __name__ == "__main__":

    class prob_arg_source:
        sigma_1 = 1
        mu_1 = 0
        sigma_2 = 1
        mu_2 = 0
        ub = 10
        total_resource = 5

    class prob_arg_target:
        sigma_1 = 2
        mu_1 = 3
        sigma_2 = 1
        mu_2 = -1
        ub = 15
        total_resource = 20


    problems_source = [problem_generator(prob_arg_source) for _ in range(1000)]
    problems_target = [problem_generator(prob_arg_target) for _ in range(100)]
    problems = (problems_source, problems_target)
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    len_x = 5
    len_lambda = 2 * len_x +1

    num_epoch = 10
    num_iteration = 5
    num_frame = 2
    
    

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    D_model_x = Discriminator(32, arg_nn)
    D_model_lambda = Discriminator(32, arg_nn)

    x_model.load_state_dict(torch.load('x_model.pth'))
    lambda_model.load_state_dict(torch.load('lambda_model.pth'))

    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.004)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)
    D_optimizer_x = torch.optim.Adam(D_model_x.parameters(), lr=0.001)
    D_optimizer_lambda = torch.optim.Adam(D_model_lambda.parameters(), lr=0.001)

    model = (x_model, lambda_model, D_model_x, D_model_lambda)
    optimizer = (x_optimizer, lambda_optimizer, D_optimizer_x, D_optimizer_lambda)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)