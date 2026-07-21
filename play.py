#!/usr/bin/python3
import torch
from state import State
from train import Net
import chess.svg
import base64
import traceback


class Valuator():
    def __init__(self):
        vals = torch.load("nets/value_net_2M.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()
        return self.model(torch.tensor(brd).float()).item()


def alpha_beta(s, v, depth, alpha, beta, maximizing_player):
    # Se abbiamo raggiunto la profondità massima o la partita è finita
    if depth == 0 or s.board.is_game_over():
        # Valutazioni estreme per evitare/cercare lo scacco matto
        if s.board.is_checkmate():
            # Se è il turno di chi massimizza, vuol dire che ha appena subìto matto
            return -10000 if maximizing_player else 10000
        # Altrimenti, delega la valutazione alla rete neurale
        return v(s)

    if maximizing_player:
        max_eval = -float('inf')
        for move in s.board.legal_moves:
            s.board.push(move)
            ev = alpha_beta(s, v, depth - 1, alpha, beta, False)
            s.board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in s.board.legal_moves:
            s.board.push(move)
            ev = alpha_beta(s, v, depth - 1, alpha, beta, True)
            s.board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

def computer_move(s, v, depth=3):
    """
    Sceglie la mossa calcolando 'depth' turni nel futuro.
    depth=2 significa: Mia mossa -> Tua risposta.
    """
    maximizing = s.board.turn == chess.WHITE
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    if maximizing:
        best_eval = -float('inf')
        for move in s.board.legal_moves:
            s.board.push(move)
            ev = alpha_beta(s, v, depth - 1, alpha, beta, False)
            s.board.pop()
            if ev > best_eval:
                best_eval = ev
                best_move = move
            alpha = max(alpha, ev)
    else:
        best_eval = float('inf')
        for move in s.board.legal_moves:
            s.board.push(move)
            ev = alpha_beta(s, v, depth - 1, alpha, beta, True)
            s.board.pop()
            if ev < best_eval:
                best_eval = ev
                best_move = move
            beta = min(beta, ev)

    if best_move:
        print(f"Turno: {'Bianco' if maximizing else 'Nero'} | Mossa scelta: {best_move} | Eval attesa: {best_eval:.3f}")
        s.board.push(best_move)

s = State()
v = Valuator()


from flask import Flask, Response, request, jsonify
app = Flask(__name__)

@app.route('/')
def hello():
    ret = open('index.html').read()
    return ret.replace('start', s.board.fen())


@app.route("/newgame")
def newgame():
    s.board.reset()
    
    # Leggiamo quale colore ha scelto il giocatore (di base il bianco)
    player_color = request.args.get('color', default='white')
    
    # Se il giocatore umano ha scelto il Nero, l'IA gioca col Bianco e deve fare la prima mossa!
    if player_color == 'black':
        computer_move(s, v)
        
    return jsonify({
        "fen": s.board.fen(),
        "eval": v(s),
        "game_over": s.board.is_game_over(),
        "result_text": get_game_result(s.board),
        "legal_moves": get_legal_moves_map(s.board)
    })


# Helper to get a clear Italian game-over reason
def get_game_result(board):
    if not board.is_game_over():
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

# Helper per mappare le mosse legali per la UI frontend
def get_legal_moves_map(board):
    moves_map = {}
    for move in board.legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        if from_sq not in moves_map:
            moves_map[from_sq] = []
        moves_map[from_sq].append(to_sq)
    return moves_map

@app.route("/get_state")
def get_state():
    return jsonify({
        "fen": s.board.fen(),
        "eval": v(s),
        "game_over": s.board.is_game_over(),
        "result_text": get_game_result(s.board),
        "legal_moves": get_legal_moves_map(s.board)
    })

@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    parsed_move = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)

    if parsed_move in s.board.legal_moves:
        move_san = s.board.san(parsed_move)
        print("human moves", move_san)
        try:
            s.board.push_san(move_san)
            computer_move(s, v)
        except Exception:
            traceback.print_exc()
    else:
        print(f"Mossa ignorata dal server (illegale): {parsed_move}")

    return jsonify({
      "fen": s.board.fen(),
      "eval": v(s),
      "game_over": s.board.is_game_over(),
      "result_text": get_game_result(s.board),
      "legal_moves": get_legal_moves_map(s.board)
    })

  print("GAME IS OVER")
  return jsonify({
    "fen": s.board.fen(),
    "eval": v(s),
    "game_over": True,
    "result_text": get_game_result(s.board),
    "legal_moves": get_legal_moves_map(s.board)
  })

if __name__ == "__main__":
    app.run(debug=True)
