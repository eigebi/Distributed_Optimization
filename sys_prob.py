import numpy as np

#test problem is simple and do not contains grouping or partitioning

#The base problem is a class that contains the objective function and the indices of variables that are involved in the function. The indices are drawn based on the global variables.
#The problem contains two operations, add and multiply. The computation result is to return the new optimization objective function, the new variable id, the graph among those variables, and the optimization direction (min/max). 
class BaseProblem:
    def __init__(self,func,varID,is_min=True):
        self.func = func
        #this ID is only used to call the global variable
        self.varID = varID
        
        self.is_min = is_min
    def __call__(self, x):
        return self.func(x)
    
    def __add__(self, other):
        return BaseProblem(lambda x: self.func(x) + other.func(x),self.varID)
    def __mul__(self, other):
        return BaseProblem(lambda x: self.func(x) * other.func(x),self.varID)
    def append(self):
        pass



def exam_func(arg):
    # we return a polynomial function give the arg as coefficients
    # the function has 3 variables with maximum degree 2
    
    return lambda x: x @ arg[:-2] @ x + arg[-1]


    




class prob:
    def __init__(self):
        self.obj = []
        self.con = []
        self.r = []


if __name__ == '__main__':
    f_s = []
    for _ in range(5):
            temp = np.random.randn(3,3)
            temp = temp @ temp.T
            f_s.append(lambda x: x @ temp @ x)
        #generate 3 integers from (1,5) without repetition
        #np.random.choice(5,3,replace=False)
    p = prob()
    for f in f_s:
        p.obj.append(BaseProblem(f,np.random.choice(5,3,replace=False)))
    c_s = []
    for i in range(5):
        p.con.append(BaseProblem(lambda x: x - np.random.randint(1,16) ,[i]))
    for i in range(5):
        p.con.append(BaseProblem(lambda x: -x ,[i]))
    p.con.append(BaseProblem(lambda x: sum(x)-30 ,np.arange(6)))
    
    for i in range(11):
        p.r.append(BaseProblem(lambda x: x , [i],is_min=False))

    z = np.random.randn(5)
    y = np.abs(np.random.randn(11))


    T = 100
    for t in range(T):
        