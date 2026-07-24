#!/usr/bin/python3
import torch
import chess
import traceback
from flask import Flask, request, jsonify
from state import State
from train import ChessResNet, encode_move
import threading


class NeuralEvaluator():
    def __init__(self, model_path):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Caricamento modello su {self.device.upper()}...")
        
        self.model = ChessResNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def evaluate_and_order(self, s):
        """
        Interroga la ResNet e restituisce:
        1. Le mosse legali ordinate per bontà (Policy Head)
        2. La valutazione della posizione ASSOLUTA (Value Head)
        """
        brd = s.serialize()
        board_tensor = torch.tensor(brd).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value_eval = self.model(board_tensor)
            
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value_eval.item()
        
        if s.board.turn == chess.BLACK:
            value = -value

        move_scores = []
        for move in s.board.legal_moves:
            idx = encode_move(move)
            score = policy_logits[idx]
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [m[0] for m in move_scores]
        
        return ordered_moves, value


def alpha_beta(s, evaluator, depth, alpha, beta, maximizing_player):
    if depth == 0 or s.board.is_game_over(claim_draw=True):
        
        if s.board.is_checkmate():
            return -(10000 + depth) if maximizing_player else (10000 + depth)

        if s.board.is_game_over(claim_draw=True):
            return 0.0
            
        _, val = evaluator.evaluate_and_order(s)
        return val

    ordered_moves, _ = evaluator.evaluate_and_order(s)

    if maximizing_player:
        max_eval = -float('inf')
        for move in ordered_moves:
            s.board.push(move)
            ev = alpha_beta(s, evaluator, depth - 1, alpha, beta, False)
            s.board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            s.board.push(move)
            ev = alpha_beta(s, evaluator, depth - 1, alpha, beta, True)
            s.board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

def computer_move(s, evaluator, depth=3):
    maximizing = s.board.turn == chess.WHITE
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    ordered_moves, current_eval = evaluator.evaluate_and_order(s)

    if maximizing:
        best_eval = -float('inf')
        for move in ordered_moves:
            s.board.push(move)
            ev = alpha_beta(s, evaluator, depth - 1, alpha, beta, False)
            s.board.pop()
            if ev > best_eval:
                best_eval = ev
                best_move = move
            alpha = max(alpha, ev)
    else:
        best_eval = float('inf')
        for move in ordered_moves:
            s.board.push(move)
            ev = alpha_beta(s, evaluator, depth - 1, alpha, beta, True)
            s.board.pop()
            if ev < best_eval:
                best_eval = ev
                best_move = move
            beta = min(beta, ev)

    if best_move:
        print(f"Turno: {'Bianco' if maximizing else 'Nero'} | Mossa scelta: {best_move} | Eval attesa: {best_eval:.3f}")
        s.board.push(best_move)


def get_game_result(board):
    if not board.is_game_over(claim_draw=True):
        return ""
    if board.is_checkmate():
        winner = "il Nero" if board.turn == chess.WHITE else "il Bianco"
        return f"Scacco Matto! Vince {winner}."
    if board.is_stalemate():
        return "Patta (Stallo)."
    if board.is_insufficient_material():
        return "Patta (Materiale Insufficiente)."
    if board.can_claim_draw():
        return "Patta (Ripetizione o 50 mosse)."
    return "Partita Terminata (Patta)."

def get_legal_moves_map(board):
    moves_map = {}
    for move in board.legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        if from_sq not in moves_map:
            moves_map[from_sq] = []
        moves_map[from_sq].append(to_sq)
    return moves_map



s = State()
v = NeuralEvaluator("nets/mini_alphazero_p913K_d6000k.pth")
game_lock = threading.Lock()


app = Flask(__name__)


@app.route('/')
def hello():
    ret = open('index.html').read()
    return ret.replace('start', s.board.fen())

@app.route("/get_state")
def get_state():
    _, current_eval = v.evaluate_and_order(s)
    return jsonify({
        "fen": s.board.fen(),
        "eval": current_eval,
        "game_over": s.board.is_game_over(claim_draw=True), 
        "result_text": get_game_result(s.board),
        "legal_moves": get_legal_moves_map(s.board)
    })

@app.route("/newgame")
def newgame():
    with game_lock:
        s.board.reset()
        player_color = request.args.get('color', default='white')
        if player_color == 'black':
            computer_move(s, v)
            
        _, current_eval = v.evaluate_and_order(s)
        
        return jsonify({
            "fen": s.board.fen(),
            "eval": current_eval,
            "game_over": s.board.is_game_over(claim_draw=True), 
            "result_text": get_game_result(s.board),
            "legal_moves": get_legal_moves_map(s.board)
        })

@app.route("/move_coordinates")
def move_coordinates():
    with game_lock:
        accepted = False
        if not s.board.is_game_over():
            source = int(request.args.get('from', default=''))
            target = int(request.args.get('to', default=''))
            promotion = True if request.args.get('promotion', default='') == 'true' else False

            parsed_move = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)

            if parsed_move in s.board.legal_moves:
                move_san = s.board.san(parsed_move)
                print("Mossa umana:", move_san)
                try:
                    s.board.push_san(move_san)
                    accepted = True
                except Exception:
                    traceback.print_exc()
            else:
                print(f"Mossa ignorata dal server (illegale): {parsed_move}")

        _, current_eval = v.evaluate_and_order(s)
        return jsonify({
            "fen": s.board.fen(),
            "eval": current_eval,
            "game_over": s.board.is_game_over(claim_draw=True), 
            "result_text": get_game_result(s.board),
            "legal_moves": get_legal_moves_map(s.board),
            "accepted": accepted
        })

@app.route("/ai_move")
def ai_move():
    with game_lock:
        if not s.board.is_game_over():
            try:
                computer_move(s, v)
            except Exception:
                traceback.print_exc()
                
        _, current_eval = v.evaluate_and_order(s)
        return jsonify({
            "fen": s.board.fen(),
            "eval": current_eval,
            "game_over": s.board.is_game_over(claim_draw=True), 
            "result_text": get_game_result(s.board),
            "legal_moves": get_legal_moves_map(s.board)
        })

if __name__ == "__main__":
    app.run(debug=True)