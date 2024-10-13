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
    
MAXVAL = 10000
class ClassicValuator():
    values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

    def __init__(self):
        pass

    def __call__(self, s):
        val = self.value(s)
        return val


    def value(self, s):
        b = s.board
        # game over values
        if b.is_game_over():
            if b.result() == "1-0":
                return MAXVAL
            elif b.result() == "0-1":
                return -MAXVAL
            else:
                return 0

        val = 0.0
        # piece values
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval
        return val

def explore_leaves(s, v):
    ret = []
    for e in s.edges():
        s.board.push(e)
        ret.append((v(s), e))
        s.board.pop()
    return ret

s = State()
v = Valuator()
#v = ClassicValuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')


def computer_minimax(s, v, depth):
    if depth == 0 or s.is_game_over():
        return v(s)


def computer_move(s, v):
    move = sorted(explore_leaves(s, v), key= lambda x:x[0], reverse=s.board.turn)
    if len(move) == 0:
        return
    print("top 3:")
    for i,m in enumerate(move[0:3]):
        print(" ", m)

    print(s.board.turn, "moving", move[0][1])
    s.board.push(move[0][1])


from flask import Flask, Response, request
app = Flask(__name__)

@app.route('/')
def hello():
    ret = open('index.html').read()
    return ret.replace('start', s.board.fen())

@app.route('/move')
def move():
    if not s.board.is_game_over():
        move = request.args.get('move', default="")
        if move is not None and move != "":
            print("io scelgo= ", move)
            s.board.push_san(move)
        computer_move(s,v)
    else:
        print("GAME IS OVER!")
    return hello()

@app.route('/board.svg')
def board():
    return Response(chess.svg.board(board = s.board), mimetype="img+xml")

@app.route('/selfplay')
def selfplay():
    ret = "<html> <head>"
    s = State()
    while not s.board.is_game_over():
        l = sorted(explore_leaves(s, v), key= lambda x:x[0], reverse=s.board.turn)
        move = l[0]
        print(move)
        s.board.push(move[1])
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(s)
    print(s.board.result())
    return ret


@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response

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
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response


if __name__ == "__main__":
    app.run(debug=True)
# if __name__ == "__main__":
#     s = State()
#     v = Valuator()
#     while not s.board.is_game_over():
#         l = sorted(explore_leaves(s, v), key= lambda x:x[0], reverse=s.board.turn)
#         move = l[0]
#         print(move)
#         s.board.push(move[1])
#     print(s.board.result())
