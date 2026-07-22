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
import pyarrow.parquet as pq
import random

class ChessDataset(Dataset):
    def __init__(self, data_source, max_rows=None, verbose=True):
        """
        data_source: può essere una stringa (percorso al file .parquet)[cite: 1] 
                     oppure direttamente un DataFrame Pandas (per lo streaming).[cite: 1]
        """
        if isinstance(data_source, str):
            if verbose:
                print(f"[{data_source}] Lettura del file Parquet dal disco...")
            df = pd.read_parquet(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source
        else:
            raise ValueError("data_source deve essere un percorso (str) o un DataFrame!")
            
        if max_rows is not None:
            df = df.head(max_rows)
            
        self.data = []
        
        iterator = df.iterrows() if not verbose else tqdm(df.iterrows(), total=len(df), desc="Preparazione Dati")
        
        for _, row in iterator:
            fen = str(row['fen'])
            turno_bianco = fen.split()[1] == 'w'
            
            colonna_mossa = 'line' if 'line' in df.columns else 'move'
            linea = str(row.get(colonna_mossa, ''))
            if not linea or len(linea) < 4 or linea == 'nan':
                continue 
            best_move = linea.split()[0]
            
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
            
        if verbose:
            print(f"Dataset pronto: {len(self.data)} posizioni caricate in RAM.")

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
    Converte un oggetto chess.Move in un indice intero compreso tra 0 e 4095.[cite: 1]
    """
    return (move.from_square * 64) + move.to_square


def decode_move(index: int, board: chess.Board = None) -> chess.Move:
    """
    Converte un indice da 0 a 4095 in un oggetto chess.Move.[cite: 1]
    """
    from_sq = index // 64
    to_sq = index % 64
    
    if board is not None:
        for move in board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                # se è una promozione, python-chess elenca per prima la promozione a Regina!)[cite: 1]
                return move
                
    # Fallback standard se non passiamo la scacchiera (nessuna promozione)[cite: 1]
    return chess.Move(from_sq, to_sq)


def get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """
    Crea una maschera booleana di 4096 elementi[cite: 1]
    """
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        idx = encode_move(move)
        mask[idx] = True
    return mask


class ResidualBlock(nn.Module):
    """
    Mantiene le dimensioni spaziali (8x8) intatte usando kernel 3x3 e padding 1.[cite: 1]
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
        p = p.view(p.size(0), -1)  # Flatten: da (Batch, 2, 8, 8) a (Batch, 128)[cite: 1]
        policy_logits = self.policy_fc(p)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten: da (Batch, 1, 8, 8) a (Batch, 64)[cite: 1]
        v = F.relu(self.value_fc1(v))
        value_eval = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value_eval
  
def validate_model(modello, val_file_path, chunk_size=100000, max_val_positions=200000, device='cpu'):
    """
    Calcola la loss su dati mai visti dal modello.[cite: 1]
    """
    print(f"\nVALIDATION sul file: {val_file_path}")
    modello.eval()
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    val_loss_tot = 0.0
    val_loss_p = 0.0
    val_loss_v = 0.0
    total_batches = 0
    posizioni_viste_val = 0
    
    parquet_file = pq.ParquetFile(val_file_path)
    
    val_rows_total = min(parquet_file.metadata.num_rows, max_val_positions) if max_val_positions else parquet_file.metadata.num_rows
    pbar_val = tqdm(total=val_rows_total, desc="Validation", unit="pos", leave=False)
    
    with torch.no_grad():
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_val_chunk = batch.to_pandas()
            val_dataset = ChessDataset(df_val_chunk, verbose=False)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            righe_saltate = len(df_val_chunk) - len(val_dataset)
            
            for board_tensor, policy_target, value_target in val_loader:
                board_tensor = board_tensor.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device)
                
                policy_logits, value_eval = modello(board_tensor)
                
                loss_p = criterion_policy(policy_logits, policy_target)
                loss_v = criterion_value(value_eval, value_target)
                
                val_loss_p += loss_p.item()
                val_loss_v += loss_v.item()
                val_loss_tot += (loss_p.item() + loss_v.item())
                total_batches += 1
                
                posizioni_batch = board_tensor.size(0)
                posizioni_viste_val += posizioni_batch
                pbar_val.update(posizioni_batch)
                
                if max_val_positions and posizioni_viste_val >= max_val_positions:
                    break
                
            if righe_saltate > 0 and posizioni_viste_val < max_val_positions:
                pbar_val.update(righe_saltate)
                
            del df_val_chunk, val_dataset, val_loader
            gc.collect()
            
            if max_val_positions and posizioni_viste_val >= max_val_positions:
                break
            
    pbar_val.close()
    modello.train()
    if total_batches == 0:
        return 0.0, 0.0, 0.0
        
    return val_loss_tot / total_batches, val_loss_p / total_batches, val_loss_v / total_batches


def train_loop(modello, train_files, val_file, chunk_size=100000, max_positions_per_epoch=2000000, epochs=2, lr=0.001, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
    print(f"Training su device: {device}")
    
    totale_righe_reali = sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)    
    totale_righe_epoch = min(totale_righe_reali, max_positions_per_epoch) if max_positions_per_epoch else totale_righe_reali
    print(f"Posizioni totali disponibili: {totale_righe_reali:,}")
    print(f"Obiettivo posizioni per Epoca: {totale_righe_epoch:,}")
    
    posizioni_per_file = max(1, totale_righe_epoch // len(train_files)) if max_positions_per_epoch else None
    if posizioni_per_file:
        print(f"Quota distribuita: il modello elaborerà fino a ~{posizioni_per_file:,} posizioni da ogni singolo shard.")

    modello = modello.to(device)
    modello.train()
    
    optimizer = optim.Adam(modello.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    history_train = []
    history_val = []
    totale_posizioni_viste_globale = 0

    for epoch in range(epochs):
        print(f"\n=== EPOCH [{epoch+1}/{epochs}] ===")
        posizioni_viste_questa_epoca = 0
        
        random.shuffle(train_files)
        
        pbar = tqdm(total=totale_righe_epoch, desc=f"Epoch {epoch+1}/{epochs}", unit="pos")
        
        for file_idx, file_path in enumerate(train_files):
            tqdm.write(f"\nApertura Shard [{file_idx+1}/{len(train_files)}]: {file_path}")
            parquet_file = pq.ParquetFile(file_path)
            
            righe_file = parquet_file.metadata.num_rows
            chunk_totali = righe_file // chunk_size
            chunk_necessari = math.ceil(posizioni_per_file / chunk_size) if posizioni_per_file else chunk_totali
            
            max_start_chunk = max(0, chunk_totali - chunk_necessari)
            start_chunk = random.randint(0, max_start_chunk) if posizioni_per_file else 0
            if start_chunk > 0:
                tqdm.write(f"\nSalto casuale: inizio la lettura dal chunk {start_chunk} (circa dalla riga {start_chunk * chunk_size:,})...")
            
            shard_loss_accum = 0.0
            shard_batches = 0
            posizioni_viste_questo_file = 0
            
            for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                if chunk_idx < start_chunk:
                    continue
                
                df_chunk = batch.to_pandas()
                
                dataset = ChessDataset(df_chunk, verbose=False)
                dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
                
                if len(dataset) == 0:
                    del df_chunk, dataset, dataloader
                    continue
                    
                righe_saltate = len(df_chunk) - len(dataset)
                
                for board_tensor, policy_target, value_target in dataloader:
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
                    
                    shard_loss_accum += loss_totale.item()
                    shard_batches += 1
                    
                    pos_batch = board_tensor.size(0)
                    posizioni_viste_questo_file += pos_batch
                    posizioni_viste_questa_epoca += pos_batch
                    totale_posizioni_viste_globale += pos_batch
                    
                    pbar.update(pos_batch)
                    pbar.set_postfix({
                        'Shard': f"{file_idx+1}/{len(train_files)}",
                        'Pol': f"{loss_p.item():.3f}",
                        'Val': f"{loss_v.item():.3f}",
                        'Tot': f"{loss_totale.item():.3f}"
                    })
                    
                    if posizioni_per_file and posizioni_viste_questo_file >= posizioni_per_file:
                        break
                    if max_positions_per_epoch and posizioni_viste_questa_epoca >= max_positions_per_epoch:
                        break
                
                if righe_saltate > 0 and (not max_positions_per_epoch or posizioni_viste_questa_epoca < max_positions_per_epoch):
                    pbar.update(righe_saltate)
                
                del df_chunk, dataset, dataloader
                gc.collect()
                
                if posizioni_per_file and posizioni_viste_questo_file >= posizioni_per_file:
                    break
                if max_positions_per_epoch and posizioni_viste_questa_epoca >= max_positions_per_epoch:
                    break
            
            if shard_batches > 0:
                history_train.append(shard_loss_accum / shard_batches)
                
                val_loss_tot, _, _ = validate_model(modello, val_file, chunk_size, max_val_positions=50000, device=device)
                history_val.append(val_loss_tot)

            if max_positions_per_epoch and posizioni_viste_questa_epoca >= max_positions_per_epoch:
                tqdm.write(f"\nRaggiunto il tetto massimo di {max_positions_per_epoch:,} posizioni per l'epoca {epoch+1}")
                break
                
        pbar.close()
                
        train_loss_media = np.mean(history_train[-len(train_files):]) if history_train else 0.0
        val_loss_media = np.mean(history_val[-len(train_files):]) if history_val else 0.0
        
        print(f"\nFine Epoch {epoch+1} / {epochs} | Posizioni totali viste: {totale_posizioni_viste_globale:,}")
        print(f"[TRAIN LOSS]: {train_loss_media:.4f}")
        print(f"[VAL LOSS]:   {val_loss_media:.4f}")
        
    os.makedirs("./nets", exist_ok=True)
    num_params = count_parameters(modello)
    params_str = f"{num_params / 1e6:.1f}M" if num_params >= 1e6 else f"{num_params / 1e3:.0f}K"

    model_filename = f"./nets/mini_alphazero_p{params_str}_d{totale_posizioni_viste_globale // 1000000}M.pth"
    torch.save(modello.state_dict(), model_filename)
    print(f"Modello salvato: {model_filename}")

    os.makedirs("./plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_train) + 1), history_train, label='Train Loss', marker='o', color='blue', linewidth=2)
    plt.plot(range(1, len(history_val) + 1), history_val, label='Validation Loss', marker='s', color='red', linewidth=2, linestyle='--')
    
    plt.title(f"Train vs Validation Loss ({totale_posizioni_viste_globale:,} posizioni viste)", fontsize=14, fontweight='bold')
    plt.xlabel("Step di Training (Shard processati)", fontsize=12)
    plt.ylabel("Loss Media", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plot_filename = f"./plots/p{params_str}_d{totale_posizioni_viste_globale // 1000}k.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico salvato in: {plot_filename}\n")


if __name__ == "__main__":
    file_parquet = sorted(glob.glob("data/*.parquet"))
    
    if len(file_parquet) < 2:
        print("Errore: Servono almeno 2 file parquet per fare Train + Validation")
    else:
        train_files = file_parquet[:-1]
        val_file = file_parquet[-1]
        
        print(f"Trovati {len(file_parquet)} file totali.")
        print(f"Training su {len(train_files)} file | Validation sull'ultimo file ({val_file})")
        
        modello = ChessResNet()
        
        train_loop(
            modello=modello,
            train_files=train_files,
            val_file=val_file,
            chunk_size=100000,
            max_positions_per_epoch=2000000,
            epochs=3
        )