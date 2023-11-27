# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque


@register_agent("heuristic_agent")
class HeuristicAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(HeuristicAgent, self).__init__()
        self.name = "heuristic_agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.max_step = None
        self.board_size = None
        self.chess_board = None
        self.end_game = False
        self.num_of_walls = None
        self.my_move_count = None
        self.max_time = 1.95
        self.start_time = None
        self.max_depth = 2
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4) (int, int, bool)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        """

        self.chess_board = chess_board

        if self.board_size is None:
            self.board_size = chess_board.shape[0]

        if self.max_step is None:
            self.max_step = max_step

        if self.board_size > 7:
            self.max_depth = 2

        self.start_time = time.time()

        best_move = self.alpha_beta(my_pos, adv_pos, self.max_depth)

        new_pos, new_dir = best_move

        time_taken = time.time() - self.start_time

        if(time_taken > self.max_time):
            print("My AI's turn took ", time_taken, "seconds")
 
        # dummy return
        return new_pos, new_dir

    
    
    def check_endgame(self,my_pos, adv_pos):
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        my_score = list(father.values()).count(p0_r)
        adv_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, my_score, adv_score
        else:
            return True, my_score, adv_score
        
    def evaluate_winner(self,my_pos, adv_pos):
        endgame, my_score, adv_score = self.check_endgame(my_pos, adv_pos)
        
        if endgame:
            if my_score == adv_score:
                return 0
            if my_score > adv_score:
                return sys.maxsize
            elif my_score < adv_score:
                return -sys.maxsize
        
        return None

    def generate_all_moves_bfs(self,my_pos, adv_pos):
        moves = []
        queue = deque([(my_pos, self.max_step)])  # Queue to store positions with remaining steps
        visited_positions = np.zeros((self.board_size, self.board_size), dtype=bool)
        visited_positions[my_pos[0]][my_pos[1]] = True

        while queue:
            current_pos, steps_remaining = queue.popleft()
            r, c = current_pos

            allowed_barriers = [i for i in range(0, 4) if not self.chess_board[r, c, i]]

            for barrier_dir in allowed_barriers:
                    potential_move = (current_pos, barrier_dir)
                    moves.append(potential_move)
   
            if steps_remaining <= 0:
                continue

            for dir, move in enumerate(self.moves):
                if(self.chess_board[r, c, dir]): continue
                if adv_pos == (r + move[0], c + move[1]): continue
                if visited_positions[r + move[0]][ c + move[1]]: continue

                my_new_pos = (r + move[0], c + move[1])
                visited_positions[r + move[0]][ c + move[1]] = True
                queue.append((my_new_pos, steps_remaining - 1))

        return moves
    

    def add_Wall(self, my_pos, dir):
        x, y = my_pos
        adj_x, adj_y = (x + self.moves[dir][0], y + self.moves[dir][1])
        self.chess_board[x, y, dir] = True
        self.chess_board[adj_x, adj_y, self.opposites[dir]] = True
        return my_pos, dir

    def remove_Wall(self, my_pos, dir):
        x, y = my_pos
        adj_x, adj_y = (x + self.moves[dir][0], y + self.moves[dir][1])
        self.chess_board[x, y, dir] = False
        self.chess_board[adj_x, adj_y, self.opposites[dir]] = False
        return my_pos, dir
    
    def alpha_beta(self,my_pos, adv_pos, depth):  
        
        best_move = None
        alpha = -sys.maxsize
        beta = sys.maxsize

        moves_max = self.generate_all_moves_bfs(my_pos, adv_pos)

        for move in moves_max:
            new_my_pos, new_dir = move
            
            self.add_Wall(new_my_pos, new_dir)
            eval_winner = self.evaluate_winner(new_my_pos, adv_pos)
            if eval_winner is None:
                score = self.minValue(new_my_pos, adv_pos, depth - 1, alpha, beta)
            else:
                score = eval_winner
            self.remove_Wall(new_my_pos, new_dir)

            if score > alpha:
                best_move = move
                alpha = score
        
        if best_move is None:
            return moves_max[-1]
        
        return best_move
    
    
    def maxValue(self,my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(my_pos, adv_pos)
        
        # Generate all possible moves for the player
        moves_max = self.generate_all_moves_bfs(my_pos, adv_pos)

        if len(moves_max) == 0:
            return self.evaluate_position(my_pos, adv_pos)

        for max_move in moves_max:
            new_my_pos, new_dir = max_move

            self.add_Wall(new_my_pos, new_dir)
            eval_winner = self.evaluate_winner(new_my_pos, adv_pos)
            if eval_winner is None:
                alpha = max(alpha,self.minValue(new_my_pos, adv_pos, depth - 1, alpha, beta))
            else:
                alpha = max(alpha,eval_winner)
                
            self.remove_Wall(new_my_pos, new_dir)

            if alpha >= beta:
                return beta
            
            if(time.time() - self.start_time > self.max_time):
                return alpha

        return alpha

    def minValue(self, my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(my_pos, adv_pos)

        #Generate all possible moves for the adversary
        moves_min = self.generate_all_moves_bfs(adv_pos, my_pos)

        if len(moves_min) == 0:
            return self.evaluate_position(my_pos, adv_pos)

        for min_move in moves_min:
            new_adv_pos, new_dir = min_move

            self.add_Wall(new_adv_pos, new_dir)
            eval_winner = self.evaluate_winner(my_pos, new_adv_pos)
            if eval_winner is None:
                beta = min(beta,self.maxValue(my_pos, new_adv_pos, depth - 1, alpha, beta))
            else:
                beta = min(beta,eval_winner)

            self.remove_Wall(new_adv_pos, new_dir)

            if alpha >= beta:
                return alpha
            
            if(time.time() - self.start_time > self.max_time):
                return beta
            
        return beta

    def count_all_moves(self, my_pos, adv_pos):
        move_count = 0
        queue = deque([(my_pos, self.max_step)])  # Queue to store positions with remaining steps
        visited_positions = np.zeros((self.board_size, self.board_size), dtype=bool)
        visited_positions[my_pos[0]][my_pos[1]] = True

        while queue:
            current_pos, steps_remaining = queue.popleft()
            r, c = current_pos

            allowed_barriers = [i for i in range(0, 4) if not self.chess_board[r, c, i]]

            for _ in allowed_barriers:
                    move_count+= 1
   
            if steps_remaining <= 0:
                continue

            for dir, move in enumerate(self.moves):
                if(self.chess_board[r, c, dir]): continue
                if adv_pos == (r + move[0], c + move[1]): continue
                if visited_positions[r + move[0]][ c + move[1]]: continue

                my_new_pos = (r + move[0], c + move[1])
                visited_positions[r + move[0]][ c + move[1]] = True
                queue.append((my_new_pos, steps_remaining - 1))

        return move_count

    def evaluate_position(self,my_pos, adv_pos):

        #Counts all max step moves
        move_count =self.count_all_moves(my_pos, adv_pos)
        adv_move_count = self.count_all_moves(adv_pos, my_pos)
          
        return move_count - adv_move_count
    
