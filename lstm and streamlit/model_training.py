import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
from sklearn.preprocessing import StandardScaler
import joblib

scaler= StandardScaler()

def make_sequence(df, seq=7):
    x= []
    target= []
    for i in range(len(df)- seq-1):
        s= df.iloc[i: i+seq, :]
        t= df.iloc[i+seq, 0]
        x.append(s)
        target.append(t)
    
    return np.array(x), np.array(target)


df= pd.read_csv("artifacts/processed/processed_train.csv", index_col= "date", parse_dates=True)
print(df.head())
print(df.info())

df_train= df.loc[:'2016-06', :]
df_val= df.loc['2016-06':, :]

df_train= pd.DataFrame(scaler.fit_transform(df_train), columns=scaler.get_feature_names_out())
df_val= pd.DataFrame(scaler.transform(df_val), columns= scaler.get_feature_names_out())

print(df_train.info())
print(df_val.info())
x_train, y_train= make_sequence(df_train)
x_val, y_val= make_sequence(df_val)

# reshape
x_train= x_train.reshape(x_train.shape[0], 7, 4)
x_val= x_val.reshape(x_val.shape[0], 7, 4)

x_train= torch.tensor(x_train, dtype= torch.float32)
y_train= torch.tensor(y_train, dtype= torch.float32)
x_val= torch.tensor(x_val, dtype= torch.float32)
y_val= torch.tensor(y_val, dtype= torch.float32)

train_dataset= TensorDataset(x_train,  y_train)
val_dataset= TensorDataset(x_val, y_val)

train_dataloader= DataLoader(train_dataset, batch_size= 32, shuffle=True)
val_dataloader= DataLoader(val_dataset, batch_size=32)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.lstm= torch.nn.LSTM(
            input_size, 
            hidden_size,
            n_layers,
            dropout= 0.3,
            batch_first=True
        )
        self.n_layers= n_layers
        self.fc= torch.nn.Linear(hidden_size, 1)
        self.hidden_size= hidden_size

    def forward(self, x):

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        x, _= self.lstm(x, (h0, c0))
        x= self.fc(x[:, -1, :])
        return x

device= "cuda" if torch.cuda.is_available() else "cpu"
model= LSTMModel(4, 110, 2).to(device)
optimizer= torch.optim.Adam(model.parameters(), lr= 4e-3)
criterion= torch.nn.MSELoss().to(device)

n_epochs= 100
for epoch in range(n_epochs):
    model.train()
    train_loss=0
    
    for x, y in train_dataloader:
        x, y= x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output= model(x)
        loss= torch.sqrt(criterion(output.view(-1), y))
        loss.backward()
        optimizer.step()

        train_loss+= loss.item()

    model.eval()
    eval_loss= 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x, y= x.to(device), y.to(device)

            output= model(x)
            loss= torch.sqrt(criterion(output.view(-1), y))
            eval_loss+= loss.item()

    print(f"Epoch: {epoch+1}/{n_epochs} Trainloss: {train_loss/len(train_dataloader)}, Testloss: {eval_loss/len(val_dataloader)}")

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")


model= joblib.load("model.pkl")
scaler= joblib.load("scaler.pkl")
test_temperature_val= pd.DataFrame([[8, 84.5, 0, 1015], 
                                   [7, 74, 2.3, 1014.6], 
                                   [8.9, 84, 3.6, 1018.78],
                                   [6.4, 86, 4, 1017.4],
                                   [7.9, 83, 3.3, 1015.4],
                                   [9.2, 78, 2.5, 1019.28],
                                   [8.8, 88, 3.8, 1017]], columns= scaler.get_feature_names_out())



df_test= pd.DataFrame(scaler.transform(test_temperature_val), columns=scaler.get_feature_names_out())
x_test= torch.tensor(df_test.values, dtype= torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output= model(x_test).detach().cpu().numpy()
    output= np.tile(output, (1, 4))
    output= scaler.inverse_transform(output)
    print(output.shape)
    ##sys.exit()
    #out= scaler.inverse_transform(output.view(-1, 1).to("cpu"))
    #print(out.view(-1).item())

    temperature= output[0,0]
    print(temperature)

