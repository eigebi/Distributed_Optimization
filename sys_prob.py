import numpy as np

#test problem is simple and do not contains grouping or partitioning

#The base problem is a class that contains the objective function and the indices of variables that are involved in the function. The indices are drawn based on the global variables.
#The problem contains two operations, add and multiply. The computation result is to return the new optimization objective function, the new variable id, the graph among those variables, and the optimization direction (min/max). 
class BaseProblem:
    def __init__(self,func,varID):
        self.func = func
        self.varID = varID
    def __call__(self, x):
        return self.func(x)
    
    def __add__(self, other):
        return BaseProblem(lambda x: self.func(x) + other.func(x),self.varID)
    def __mul__(self, other):
        return BaseProblem(lambda x: self.func(x) * other.func(x),self.varID)



def exam_func(arg):
    # we return a polynomial function give the arg as coefficients
    # the function has 3 variables with maximum degree 2
    
    return lambda x: arg[0] + arg[1]*x[0] + arg[2]*x[1] + arg[3]*x[2] + arg[4]*x[0]*x[1] + arg[5]*x[1]*x[2] + arg[6]*x[0]*x[2] + arg[7]*x[0]*x[0] + arg[8]*x[1]*x[1] + arg[9]*x[2]*x[2]


    





class test_prob:
    def __init__(self):
        self.a = 1
        self.b = 2




class prob:
    def __init__(self):
        pass


if __name__ == '__main__':
   f_s = []
   for _ in range(5):
        temp = np.randn((3,3))
        temp = temp @ temp.T
        f_s.append(lambda x: x @ temp @ x)
    #generate 3 integers from (1,5) without repetition
    #np.random.choice(5,3,replace=False)
   
   for f in f_s:
       