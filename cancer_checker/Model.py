import torch
from torch import nn
from torch.nn.modules.pooling import MaxPool2d
class CNNmodel(nn.Module):
  def __init__(self,inpshape:int,hiddenunits:int,outputshape:int):
    super().__init__()
    self.conv_block1=nn.Sequential(
      nn.Conv2d(in_channels=inpshape,out_channels=hiddenunits,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hiddenunits,out_channels=hiddenunits,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)  
       )
    self.conv_block2=nn.Sequential(
      nn.Conv2d(in_channels=hiddenunits,out_channels=hiddenunits,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hiddenunits,out_channels=hiddenunits,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)  
       )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hiddenunits*16*16,out_features=outputshape)
    )
    
  def forward(self,x:torch.Tensor):
     #print(f"Input to layer 1 : {x.shape}")
     x= self.conv_block1(x.to(torch.float))
     #print(f"output layer 1 : {x.shape}")
     x=self.conv_block2(x)
     #print(f"output layer 2 : {x.shape}")
     x=self.classifier(x)
     #print(f"classifier : {x.shape}")
     return x
  
model0=CNNmodel(3,16,7).to('cpu')
print(model0.load_state_dict(torch.load('./cancer_checker/Best_score_64x64_79.pt',map_location=torch.device('cpu'))))
#print(model0.state_dict())