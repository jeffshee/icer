import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def read_from_excel(file_dir):
    if file_dir is None:
        file_dir="result1.csv"
    data=pd.read_csv(file_dir)

    ##change the data to a list
    e_list=data["prediction"].values.tolist()
    m_list=data["mouse opening"].astype('float32').values.tolist()
    m_list=[round(i, 3) for i in m_list]
    g_list=data["gesture"].astype('float32').values.tolist()
    p_list=data["points"].astype('float32').values.tolist()
    int_emo=[]

    ##change the emotions to numbers
    for i in range(len(e_list)):
        if e_list[i]=="Positive":
            int_emo.append(0)
        elif e_list[i]=="Normal":
            int_emo.append(1)
        elif e_list[i]=="Negative":
            int_emo.append(2)
        else:
            int_emo.append(3)
    e_tensor = torch.from_numpy(np.array(int_emo)).unsqueeze(1).float()
    g_tensor = torch.from_numpy(np.array(g_list)).unsqueeze(1).float()
    m_tensor = torch.from_numpy(np.array(m_list)).unsqueeze(1).float()
    p_tensor = torch.from_numpy(np.array(p_list)).unsqueeze(1).float() ## one person's points

    input_tensor=torch.cat((e_tensor, g_tensor, m_tensor), 1)

    return input_tensor, p_tensor


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

lstm_model = LstmRNN(3, 16, 1, num_layers=1)

input,point=read_from_excel("output_0309-4/test.csv")
rsz_tensor=input.reshape(-1,5,3)
p_tensor=point.reshape(-1,5,1)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

max_epochs = 10000
for epoch in range(max_epochs):
    output = lstm_model(rsz_tensor)*12.0
    loss = loss_function(output, p_tensor)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if loss.item() < 2 + 1e-4: ## threshold value can be changed
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch+1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))

##save the weight
torch.save(lstm_model.state_dict(),"lstm.pth")