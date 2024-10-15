# A place for test
from models.model import *
net=Vgg16_net()
a=torch.randn(64,3,150,150)
print(net(a))