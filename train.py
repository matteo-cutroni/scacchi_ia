#!/usr/bin/python3
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import chess
from state import State
import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import gc

class ChessDataset(Dataset):
    def __init__(self, parquet_path, max_rows=100000):
        """
        Carica e processa il file Parquet.
        """
        df = pd.read_parquet(parquet_path)
        if max_rows is not None:
            df = df.head(max_rows)
        
        self.data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparazione Dataset"):
            fen = str(row['fen'])
            turno_bianco = fen.split()[1] == 'w'
            
            # Policy Head Target
            colonna_mossa = 'line' if 'line' in df.columns else 'move'
            linea = str(row.get(colonna_mossa, ''))
            if not linea or len(linea) < 4 or linea == 'nan':
                continue 
            best_move = linea.split()[0]
            
            # Value Head Target
            value = 0.0
            if 'cp' in df.columns and 'mate' in df.columns:
                if pd.notna(row['mate']):
                    mate_in = int(row['mate'])
                    vince_di_turno = 1.0 if mate_in > 0 else -1.0
                    value = vince_di_turno if turno_bianco else -vince_di_turno
                elif pd.notna(row['cp']):
                    cp = float(row['cp'])
                    cp_assoluto = cp if turno_bianco else -cp
                    value = math.tanh(cp_assoluto / 400.0)
            elif 'eval' in df.columns:
                valutazione = str(row['eval'])
                if '#' in valutazione:
                    vince_di_turno = 1.0 if not valutazione.startswith('-') else -1.0
                    value = vince_di_turno if turno_bianco else -vince_di_turno
                else:
                    try:
                        cp = float(valutazione)
                        cp_assoluto = cp if turno_bianco else -cp
                        value = math.tanh(cp_assoluto / 400.0)
                    except ValueError:
                        continue
                        
            self.data.append((fen, best_move, value))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, mossa_str, risultato = self.data[idx]
        
        board = chess.Board(fen)
        stato = State(board)
        board_tensor = torch.from_numpy(stato.serialize())
        
        mossa_reale = chess.Move.from_uci(mossa_str)
        policy_target = encode_move(mossa_reale)
        value_target = torch.tensor([risultato], dtype=torch.float32)
        
        return board_tensor, policy_target, value_target

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def encode_move(move: chess.Move) -> int:
    """
    Converte un oggetto chess.Move in un indice intero compreso tra 0 e 4095.
    """
    return (move.from_square * 64) + move.to_square


def decode_move(index: int, board: chess.Board = None) -> chess.Move:
    """
    Converte un indice da 0 a 4095 in un oggetto chess.Move.
    """
    from_sq = index // 64
    to_sq = index % 64
    
    if board is not None:
        for move in board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                # se è una promozione, python-chess elenca per prima la promozione a Regina!)
                return move
                
    # Fallback standard se non passiamo la scacchiera (nessuna promozione)
    return chess.Move(from_sq, to_sq)


def get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """
    Crea una maschera booleana di 4096 elementi
    """
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        idx = encode_move(move)
        mask[idx] = True
    return mask


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
        
        self.conv_in = nn.Conv2d(18, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten: da (Batch, 2, 8, 8) a (Batch, 128)
        policy_logits = self.policy_fc(p)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten: da (Batch, 1, 8, 8) a (Batch, 64)
        v = F.relu(self.value_fc1(v))
        value_eval = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value_eval
  
def train_loop(modello, parquet_files, epochs=3, batch_size=128, max_rows_per_file=None, lr=0.001, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"Training su device: {device}")

    modello = modello.to(device)
    modello.train()
    
    optimizer = optim.Adam(modello.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    history_policy = []
    history_value = []
    history_total = []
    totale_posizioni_viste = 0

    for epoch in range(epochs):
        print(f"\n=== EPOCH [{epoch+1}/{epochs}] ===")
        
        for file_idx, file_path in enumerate(parquet_files):
            print(f"\nCaricamento Shard [{file_idx+1}/{len(parquet_files)}]: {file_path}")
            
            dataset = ChessDataset(file_path, max_rows=max_rows_per_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            totale_posizioni_viste += len(dataset)
            
            loss_p_accum = 0.0
            loss_v_accum = 0.0
            loss_tot_accum = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} | Shard {file_idx+1}", unit="batch")
            
            for board_tensor, policy_target, value_target in pbar:
                board_tensor = board_tensor.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device)
                
                optimizer.zero_grad()
                policy_logits, value_eval = modello(board_tensor)
                
                loss_p = criterion_policy(policy_logits, policy_target)
                loss_v = criterion_value(value_eval, value_target)
                loss_totale = loss_p + loss_v
                
                loss_totale.backward()
                optimizer.step()
                
                loss_p_accum += loss_p.item()
                loss_v_accum += loss_v.item()
                loss_tot_accum += loss_totale.item()
                
                pbar.set_postfix({
                    'L_Pol': f"{loss_p.item():.4f}",
                    'L_Val': f"{loss_v.item():.4f}",
                    'L_Tot': f"{loss_totale.item():.4f}"
                })
                
            n_batches = len(dataloader)
            if n_batches > 0:
                history_policy.append(loss_p_accum / n_batches)
                history_value.append(loss_v_accum / n_batches)
                history_total.append(loss_tot_accum / n_batches)
            
            # PULIZIA MEMORIA: cancelliamo dataset e dataloader per far posto al prossimo file
            del dataset
            del dataloader
            gc.collect()
                    
    os.makedirs("./nets", exist_ok=True)
    num_params = count_parameters(modello)
    params_str = f"{num_params / 1e6:.1f}M" if num_params >= 1e6 else f"{num_params / 1e3:.0f}K"
    pos_str = f"{totale_posizioni_viste // 1000}k" if totale_posizioni_viste >= 1000 else str(totale_posizioni_viste)
    
    model_filename = f"./nets/mini_alphazero_p{params_str}_d{pos_str}_ep{epochs}.pth"
    torch.save(modello.state_dict(), model_filename)
    print(f"Modello salvato con successo: {model_filename}")

    os.makedirs("./plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(history_total, label='Loss Totale', color='purple', linewidth=2)
    plt.plot(history_policy, label='Loss Policy (CrossEntropy)', color='blue', linestyle='--')
    plt.plot(history_value, label='Loss Value (MSE)', color='orange', linestyle='--')
    
    plt.title(f"Training su {len(parquet_files)} Shard ({totale_posizioni_viste:,} posizioni viste)", fontsize=14, fontweight='bold')
    plt.xlabel("Numero di Shard processati nel tempo", fontsize=12)
    plt.ylabel("Loss Media per Shard", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plot_filename = f"./plots/training_p{params_str}_d{pos_str}_ep{epochs}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGrafico Losses salvato in: {plot_filename}\n")


if __name__ == "__main__":
    
    file_parquet = sorted(glob.glob("data/*.parquet"))    
    modello = ChessResNet()

    train_loop(
        modello=modello, 
        parquet_files=file_parquet, 
        epochs=5, 
        batch_size=128,
        max_rows_per_file=100000
    )
