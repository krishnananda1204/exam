import numpy as np
A=np.random.randint(10,size=(3,3))
B=np.random.randint(10,size=(3,3))
print("AAT-2BBT\n",np.subtract((A*(np.transpose(A))),(2*B)*(np.transpose(B))))