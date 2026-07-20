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

def explore_leaves(s, v):
    ret = []
    for e in s.edges():
        s.board.push(e)
        ret.append((v(s), e))
        s.board.pop()
    return ret

s = State()
v = Valuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')


def computer_move(s, v):
    move = sorted(explore_leaves(s, v), key= lambda x:x[0], reverse=s.board.turn)
    if len(move) == 0:
        return
    print("top 3:")
    for i,m in enumerate(move[0:3]):
        print(" ", m)

    print(s.board.turn, "moving", move[0][1])
    s.board.push(move[0][1])


from flask import Flask, Response, request, jsonify
app = Flask(__name__)

@app.route('/')
def hello():
    ret = open('index.html').read()
    return ret.replace('start', s.board.fen())


@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response

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

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
        
    return jsonify({
      "fen": s.board.fen(),
      "eval": v(s),
      "game_over": s.board.is_game_over(),
      "result_text": get_game_result(s.board)
    })

  print("GAME IS OVER")
  return jsonify({
    "fen": s.board.fen(),
    "eval": v(s),
    "game_over": True,
    "result_text": get_game_result(s.board)
  })


if __name__ == "__main__":
    app.run(debug=True)
