#!/usr/bin/python3
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class ChessValueDataset(Dataset):
    def __init__(self):
        data = np.load("processed/dataset_2M.npz")
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print("loaded", self.X.shape, self.Y.shape)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class ResidualBlock(nn.Module):
    """
    Mantiene le dimensioni spaziali (8x8) intatte usando kernel 3x3 e padding 1.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        return out


class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=5, num_filters=64):
        super(ChessResNet, self).__init__()
        
        # Riceve i 18 canali del nostro State e li espande a num_filters (es. 64)
        self.conv_in = nn.Conv2d(18, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # 'num_res_blocks' blocchi residui uno dopo l'altro
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # --- POLICY HEAD ---
        # Comprime a 2 canali e poi proietta nello spazio delle 4096 mosse
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # --- VALUE HEAD ---
        # Comprime a 1 canale, processa linearmente e restituisce un numero da -1 a 1
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # blocco iniziale
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # blocchi residui
        for block in self.res_blocks:
            x = block(x)
            
        # 3. Ramo Policy (Output: Logits grezzi per 4096 mosse)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten: da (Batch, 2, 8, 8) a (Batch, 128)
        policy_logits = self.policy_fc(p)
        
        # 4. Ramo Value (Output: Scalare tra -1 e 1)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten: da (Batch, 1, 8, 8) a (Batch, 64)
        v = F.relu(self.value_fc1(v))
        value_eval = torch.tanh(self.value_fc2(v))  # Tanh forza l'uscita esattamente tra -1 e 1
        
        return policy_logits, value_eval
  
if __name__ == "__main__":
    if torch.cuda.is_available():
      device = torch.device('cuda')
      torch.cuda.set_device(0)
    else:
      device = torch.device('cpu')

    print('Using PyTorch version:', torch.__version__, ' Device:', device)

    chess_dataset = ChessValueDataset()
    train_loader = DataLoader(chess_dataset, batch_size=1024, shuffle=True)
    model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    
    model.train()

    for epoch in range(50):
      all_loss = 0
      num_loss = 0

      for batch_idx, (data, target) in enumerate(train_loader):
        target = target.unsqueeze(-1)
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()

        optimizer.zero_grad()
        output = model(data)

        loss = floss(output, target)
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        num_loss += 1

      print(f"{epoch}: {all_loss/num_loss}")

      torch.save(model.state_dict(), "nets/value_net_2M.pth")
