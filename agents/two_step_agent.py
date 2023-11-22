# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque


@register_agent("two_step_agent")
class TwoStepAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(TwoStepAgent, self).__init__()
        self.name = "two_step_agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.max_step = None
        self.board_size = None
        self.end_game = True
        self.chess_board = None
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

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.

        if self.board_size is None:
            self.board_size = chess_board.shape[0]

        start_time = time.time()

        if self.max_step is None:
            self.max_step = max_step

        self.chess_board = chess_board

        best_move = self.alpha_beta(my_pos, adv_pos, 3)

        new_pos, new_dir = best_move

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

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


    def generate_all_moves_bfs(self,my_pos, adv_pos, max_step, all_moves):

        queue = deque([(my_pos, max_step)])  # Queue to store positions with remaining steps
        visited_positions = set()  # To keep track of visited positions
        visited_positions.add(my_pos)

        while queue:
            current_pos, steps_remaining = queue.popleft()

            r, c = current_pos

            allowed_barriers = [i for i in range(0, 4) if not self.chess_board[r, c, i]]


            for barrier_dir in allowed_barriers:
                    potential_move = (current_pos, barrier_dir)
                    all_moves.append(potential_move)
   
            if steps_remaining <= 0:
                continue

            allowed_dirs = [
                d
                for d in range(0, 4)
                if not self.chess_board[r, c, d] and 
                not adv_pos == (r + self.moves[d][0], c + self.moves[d][1])
                and not (r + self.moves[d][0], c + self.moves[d][1]) in visited_positions
            ]

            for allowed in allowed_dirs:
                m_r, m_c = self.moves[allowed]
                my_new_pos = (r + m_r, c + m_c)
                visited_positions.add(my_new_pos)
                queue.append((my_new_pos, steps_remaining - 1))

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
        
        moves_max = []
        best_move = None
        best_score = -sys.maxsize

        self.generate_all_moves_bfs(my_pos, adv_pos, self.max_step, moves_max)

        for move in moves_max:
            new_my_pos, new_dir = move
            x, y = new_my_pos
            
            self.add_Wall(new_my_pos, new_dir)
            eval_winner = self.evaluate_winner(new_my_pos, adv_pos)
            if eval_winner is None:
                score = self.minValue(new_my_pos, adv_pos, depth - 1, -sys.maxsize, sys.maxsize)
            else:
                score = eval_winner
            self.remove_Wall(new_my_pos, new_dir)

            if score > best_score:
                best_move = move
                best_score = score
            
    
        if best_move is None:
            return moves_max[-1]
        
        return best_move
    
    
    def maxValue(self,my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(my_pos, adv_pos)
        
        moves_max = []

        # Generate all possible moves for the player
        self.generate_all_moves_bfs(my_pos, adv_pos, self.max_step, moves_max)

        if len(moves_max) == 0:
            return self.evaluate_position(my_pos, adv_pos)

        for move in moves_max:
            new_my_pos, new_dir = move
            x, y = new_my_pos

            self.add_Wall(new_my_pos, new_dir)
            eval_winner = self.evaluate_winner(new_my_pos, adv_pos)
            if eval_winner is None:
                alpha = max(alpha,self.minValue(new_my_pos, adv_pos, depth - 1, alpha, beta))
            else:
                alpha = max(alpha,eval_winner)
            self.remove_Wall(new_my_pos, new_dir)

            if alpha >= beta:
                return beta

        return alpha

    def minValue(self, my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(my_pos, adv_pos)
        
        moves_min = []

        #Generate all possible moves for the adversary
        self.generate_all_moves_bfs(adv_pos, my_pos, self.max_step, moves_min)

        if len(moves_min) == 0:
            return self.evaluate_position(my_pos, adv_pos)

        for move in moves_min:
            new_adv_pos, new_dir = move
            x, y = new_adv_pos

            self.add_Wall(new_adv_pos, new_dir)
            eval_winner = self.evaluate_winner(my_pos, new_adv_pos)
            if eval_winner is None:
                beta = min(beta,self.maxValue(my_pos, new_adv_pos, depth - 1, alpha, beta))
            else:
                beta = min(beta,eval_winner)
            self.remove_Wall(new_adv_pos, new_dir)

            if beta <= alpha:
                return alpha
            
        return beta

    def evaluate_position(self,my_pos, adv_pos):
        my_x, my_y = my_pos
        adv_x, adv_y = adv_pos

        moves_min = []

        #Generate all possible moves for the adversary
        self.generate_all_moves_bfs(adv_pos, my_pos, self.max_step, moves_min)

        my_moves = len(moves_min)
        
        count_walls = 0
        for wall in self.chess_board[my_x, my_y]:
            if wall:
                count_walls += 1
        
        point_modifier = 1
        if count_walls == 3:
            point_modifier = -10
 

          
        return my_moves + point_modifier
    
