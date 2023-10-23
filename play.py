#!/usr/bin/python3
import torch
from state import State
from train import Net
import chess.svg

class Valuator():
    def __init__(self):
        vals = torch.load("nets/value_net_10k.pth", map_location=lambda storage, loc: storage)
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
    return ret

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
