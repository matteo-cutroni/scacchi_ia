import chess
import numpy as np
import torch

class State:
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def serialize(self):
        """
        Trasforma la scacchiera in un tensore 3D di forma (18, 8, 8)
        pronto per essere consumato dalla ResNet.
        """
        # Inizializziamo una matrice di zeri 18x8x8
        bstate = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Mappatura dei pezzi per indice del canale
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # 1. Posizionamento dei pezzi (Canali 0-11)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                # Coordinate sulla griglia 8x8
                row, col = i // 8, i % 8
                idx = piece_map[piece.piece_type]
                
                # Se il pezzo è nero, lo mettiamo nei canali 6-11
                if piece.color == chess.BLACK:
                    idx += 6
                
                bstate[idx, row, col] = 1.0

        # 2. Turno di gioco (Canale 12)
        if self.board.turn == chess.WHITE:
            bstate[12, :, :] = 1.0
            
        # 3. Diritti di Arrocco (Canali 13-16)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            bstate[13, :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            bstate[14, :, :] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            bstate[15, :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            bstate[16, :, :] = 1.0
            
        # 4. En Passant (Canale 17)
        if self.board.ep_square is not None:
            row, col = self.board.ep_square // 8, self.board.ep_square % 8
            bstate[17, row, col] = 1.0
            
        return bstate

    def to_tensor(self, device='cpu'):
        # Converte in tensore PyTorch e aggiunge la dimensione del Batch (1, 18, 8, 8)
        tensor = torch.from_numpy(self.serialize()).to(device)
        return tensor.unsqueeze(0)