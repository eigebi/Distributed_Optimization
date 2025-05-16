import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

np.random.seed(1006)

#test problem is simple and do not contains grouping or partitioning

#The base problem is a class that contains the objective function and the indices of variables that are involved in the function. The indices are drawn based on the global variables.
#The problem contains two operations, add and multiply. The computation result is to return the new optimization objective function, the new variable id, the graph among those variables, and the optimization direction (min/max). 
class BaseProblem:
    def __init__(self,func,varID,is_min=True):
        self.func = func
        #this ID is only used to call the global variable
        self.varID = np.array(varID)
        self.gamma = np.random.randn(len(varID))
        self.is_min = is_min
        self.x_next = []
    def __call__(self, x):
        return self.func(x)
    
    def __add__(self, other):
        return BaseProblem(lambda x: self.func(x) + other.func(x),self.varID)
    def __mul__(self, other):
        return BaseProblem(lambda x: self.func(x) * other.func(x),self.varID)
    def append(self):
        pass

    def optimize(self,x_global):
        rho = 1
        x_i = x_global[self.varID]
        f_i = lambda x: self.func(x)+self.gamma@(x-x_i)+rho/2*np.linalg.norm(x-x_i)**2
        if self.is_min:
            x = minimize(f_i,np.zeros(len(self.varID)))
        return



class prob:
    def __init__(self):
        self.obj = []
        self.con = []
        self.r = []


class problem_generator(prob):
    def __init__(self, prob_arg, bounded = False):
        super(problem_generator,self).__init__()
        num_o= 5
        self.num_o = num_o
        self.bounded = bounded
        self.prob_arg = prob_arg
        f_s = []
        self.jac = []
        for _ in range(num_o):
            temp = self.prob_arg.sigma_1*np.random.randn(3,3) + self.prob_arg.mu_1
            temp = temp @ temp.T
            temp2 = self.prob_arg.sigma_2*np.random.randn(3) + self.prob_arg.mu_2

            f_s.append(lambda x: x @ temp @ x + temp2 @ x+0)
            self.jac.append(lambda x: 2*temp @ x + temp2)

        for f in f_s:
            var_select = np.random.choice(num_o,3,replace=False)
            self.obj.append(BaseProblem(f,var_select))
        u_b = np.zeros(num_o+1)
        for i in range(num_o): 
            u_bound = np.random.randint(1,self.prob_arg.ub)#16
            self.con.append(BaseProblem(lambda x: x - u_bound ,[i]))
            u_b[i]=u_bound
        for i in range(num_o):
            self.con.append(BaseProblem(lambda x: -x ,[i]))
        u_b[-1] = np.array([self.prob_arg.total_resource],dtype=np.float32)
        self.con.append(BaseProblem(lambda x: np.array([np.sum(x)-u_b[-1]]) ,np.arange(num_o)))
        self.u_b = u_b
        
        
        for i in range(2*num_o+1):
            self.r.append(BaseProblem(lambda x: x , [i],is_min=False))

    def __call__(self,X,R):
        batch_size = X.shape[0]
        result = np.zeros(batch_size)
        if self.bounded:
            for i in range(batch_size):
                result[i] = np.sum([self.obj[k](X[i,self.obj[k].varID]) for k in range(self.num_o)]) + np.sum(np.multiply(R[i,:],np.array([np.array(self.con[-1](X[i,self.con[-1].varID]))])[0]))
        else:
            for i in range(batch_size):
                result[i] = np.sum([self.obj[k](X[i,self.obj[k].varID]) for k in range(self.num_o)]) + np.sum(np.multiply(R[i,:],np.array([np.array(self.con[k](X[i,self.con[k].varID])) for k in range(2*self.num_o+1)])[0]))
        return result.reshape(-1,1)
    
    def solve(self):
        f = lambda x: sum([self.obj[k](x[self.obj[k].varID]) for k in range(self.num_o)])   

        x0 = np.random.randn(self.num_o)
        A = np.row_stack((np.eye(self.num_o),np.ones((1,self.num_o))))
        constraint = LinearConstraint(A,0,self.u_b)
        res = minimize(f,x0,constraints=constraint)
        return res
    
    def gradient_x(self,x,r):
        grad_x = np.zeros([x.shape[0],self.num_o],dtype=np.float32)
        for k in range(x.shape[0]):
            g = np.sum(x[k,:])
            penalty = 0
            if g>0:
                penalty = 5 * g


            for i in range(self.num_o):
                grad_x[k,self.obj[i].varID] += self.jac[i](x[k,self.obj[i].varID])
            if self.bounded:
                grad_x[k,:] = grad_x[k,:] + r[k,-1] #+ penalty
            else:
                grad_x[k,:] = grad_x[k,:] + r[k,:self.num_o] - r[k,self.num_o:-1] + r[k,-1] #+ penalty
        return grad_x
    
    def gradient_x_penalty(self,x,r):
        grad_x = np.zeros([x.shape[0],self.num_o],dtype=np.float32)
        for k in range(x.shape[0]):
            g = np.sum(x[k,:])
            penalty = 0
            if g>0:
                penalty = 5 * g
            
            for i in range(self.num_o):
                grad_x[k,self.obj[i].varID] += self.jac[i](x[k,self.obj[i].varID])
            if self.bounded:
                grad_x[k,:] = grad_x[k,:] + r[k,-1] + penalty
            else:
                grad_x[k,:] = grad_x[k,:] + r[k,:self.num_o] - r[k,self.num_o:-1] + r[k,-1]
        return grad_x
    
    def gradient_lambda(self,x):
        grad_lambda = np.zeros(2*self.num_o+1,dtype=np.float32)
        grad_lambda = np.array([self.con[k](x[0,self.con[k].varID]) for k in range(2*self.num_o+1)])
        if self.bounded:
            return grad_lambda[-1].reshape(1,-1)
        else:
            return grad_lambda.reshape(1,-1)
    
    def objective(self, x):
        return np.sum([self.obj[k](x[0,self.obj[k].varID]) for k in range(self.num_o)])
    
    def check_con(self,x):
        return np.array([self.con[k](x[0,self.con[k].varID]) for k in range(2*self.num_o+1)])
    
class Distributed_Problems():
    def __init__(self, prob_arg):
        self.problems = [problem_generator(prob_arg) for _ in range(3)]
        self.overall_ub = prob_arg.overall_ub

        
        
    
    



        

if __name__ == '__main__':

    class prob_arg:
        sigma_1 = 1
        mu_1 = 0
        sigma_2 = 1
        mu_2 = -5
        ub = 10
        total_resource = 26
        overall_ub = 5

    num_agents = 5
    problems = [problem_generator(prob_arg) for _ in range(num_agents)]
    # initial x and lambda,global lambda
    x = np.random.randn(num_agents,5)
    _x = np.zeros_like(x)
    r = np.random.randn(num_agents,11)
    r = np.abs(r)
    _r = np.zeros_like(r)
    num_global_cons = 2
    r_global = np.random.randn(num_global_cons)
    r_global = np.abs(r_global)
    y = np.random.randn(num_global_cons)
    y = np.abs(y)
    z = np.random.randn(num_global_cons)
    z = np.abs(z)

    # local x, global dual
    beta = 50
    # local lambda
    rho = 50
    # global consensus
    gamma = 20
    
    
    obj = []
    global_dual_value = []
    global_constraint = []
    for iter in range(10000):
        # local update
        for n_p, prob in enumerate(problems):
            grad_x = prob.gradient_x(x[n_p].reshape(1,-1),r[n_p].reshape(1,-1))
            grad_x = grad_x.squeeze(0)
            # add global part
            if n_p <= 2:
                grad_x = grad_x + r_global[0]
            if n_p >= 2:
                grad_x = grad_x + r_global[1]
            
            _x[n_p] = x[n_p] - grad_x/beta

            grad_lambda = prob.gradient_lambda(_x[n_p].reshape(1,-1))
            grad_lambda = grad_lambda.squeeze(0)
            _r[n_p] = np.maximum(r[n_p] + grad_lambda / rho, 0)
            # add global part
            # pay attention ot dimension
            grad_global_lambda = np.sum(x[:3])-prob_arg.overall_ub
            grad_global_lambda_2 = np.sum(x[2:])-prob_arg.overall_ub
            _r_global = np.maximum(r_global + (np.array([grad_global_lambda,grad_global_lambda_2])-y)/rho,0)

        print('obj: ',np.sum([problems[i].objective(_x[i].reshape(1,-1)) for i in range(len(problems))]))
        obj.append(np.sum([problems[i].objective(_x[i].reshape(1,-1)) for i in range(len(problems))]))
        global_dual_value.append(r_global)
        global_constraint.append(np.sum(_x)-prob_arg.overall_ub)
        # global update
        _y = np.maximum(y + rho * (_r_global - z),0)
        _z = (gamma * z + rho * r_global + y)/ (gamma + rho)
        # update x, r, r_global, y, z
        x = _x
        r = _r
        r_global = _r_global
        y = _y
        z = _z
    print("overall resource: ", np.sum(x[:3]), np.sum(x[2:]))
    print('global dual: ',r_global)
    print('optimal: ',[problems[i].objective(problems[i].solve().x.reshape(1,-1)) for i in range(len(problems))])
    x0 = np.random.randn(num_agents*5)  
    f = lambda x: sum([problems[i].objective(x[i*5:(i+1)*5].reshape(1,-1)) for i in range(num_agents)])
    
    cons = []
    A = np.row_stack((np.eye(5),np.ones((1,5))))
    _A = block_diag(*[A for _ in range(num_agents)])
    A = np.concatenate([_A,np.ones((1,num_agents*5,))],axis=0)
    ub = np.concatenate([problems[i].u_b for i in range(num_agents)]).reshape(-1,1)
    ub = np.concatenate([ub,np.array([prob_arg.overall_ub]).reshape(1,1)],axis=0)
    constraint = LinearConstraint(A,0,ub.squeeze(1))
    res = minimize(f,x0,constraints=constraint)
    print('optimal obj: ',res.fun)
    #np.save('distributed_obj.npy',np.array(obj))
    #np.save('distributed_dual.npy',np.array(global_dual_value))
    #np.save('distributed_constraint.npy',np.array(global_constraint))
  
    
    
