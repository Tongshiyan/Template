from models.model import *
class LSTM_Fea(nn.Module):
    def __init__(self):
        super(LSTM_Fea, self).__init__()
        self.model=nn.LSTM(input_size=1,hidden_size=16,num_layers=3)
    def forward(self,x):
        output,(_,_)=self.model(x)
        return output.permute(0,2,1)
x=torch.randn(64,2560,1)
mode=LSTM_Fea()
print(mode(x).shape)