import chess

class State:
    def __init__(self):
        self.board = chess.Board()

    def serialize(self):
        pass

    def edges(self):
        return list(self.board.legal_moves)
        
    def value(self):
        #TODO: qui va la rete neurale
        return 1
    

if __name__ == "__main__":
    s = State()
    print(s.edges())