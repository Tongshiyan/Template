# A place for test
import torch
from util.metrics import metric
a=torch.tensor([[0.14,0.86],[0.25,0.75]]).float()
b=torch.tensor([[0,1],[0,1]]).float()
print(a,b)
c=torch.nn.CrossEntropyLoss()
print(c(a,b))