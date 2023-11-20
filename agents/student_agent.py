# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "student_agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.max_step = None
        self.board_size = None
        self.num_of_walls = None
        self.early_game_max_step = True

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

        if self.num_of_walls is None:
            self.num_of_walls = self.count_game_walls(chess_board)

        if self.board_size is None:
            self.board_size = chess_board.shape[0]

        start_time = time.time()

        if self.max_step is None:
            self.max_step = max_step

        if self.num_of_walls > (self.board_size)**2:
            self.early_game_max_step = False

        best_move = self.alpha_beta(chess_board, my_pos, adv_pos, 2)

        new_pos, new_dir = best_move

        self.num_of_walls += 2

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return new_pos, new_dir
    
    def count_game_walls(self, chess_board):
        walls = 0
        for row in chess_board:
            for cell in row:
                for wall in cell:
                    if wall:
                        walls += 1
        return walls
    
    def check_endgame(self, chess_board, my_pos, adv_pos):
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
                    if chess_board[r, c, dir + 1]:
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
        return True, my_score, adv_score

    def generate_all_moves_bfs(self, chess_board, my_pos, adv_pos, max_step, all_moves):

        queue = deque([(my_pos, max_step)])  # Queue to store positions with remaining steps
        visited_positions = set()  # To keep track of visited positions

        while queue:
            current_pos, steps_remaining = queue.popleft()
            if steps_remaining <= 0 or current_pos in visited_positions:
                continue

            visited_positions.add(current_pos)

            r, c = current_pos

            allowed_dirs = [
                d
                for d in range(0, 4)
                if not chess_board[r, c, d] and 
                not adv_pos == (r + self.moves[d][0], c + self.moves[d][1])
            ]

            for allowed in allowed_dirs:
                m_r, m_c = self.moves[allowed]
                my_new_pos = (r + m_r, c + m_c)

                allowed_barriers = [i for i in range(0, 4) if not chess_board[r + m_r, c + m_c, i]]

                for barrier_dir in allowed_barriers:
                    potential_move = (my_new_pos, barrier_dir)
                    all_moves.append(potential_move)

                # Instead of calling the function recursively, add the new position to the queue
                queue.append((my_new_pos, steps_remaining - 1))
    

    def generate_max_moves_bfs(self, chess_board, my_pos, adv_pos, max_step, final_positions):
        queue = deque([(my_pos, max_step)])  # Queue to store positions with remaining steps
        visited_positions = set()  # To keep track of visited positions

        while queue:
            current_pos, steps_remaining = queue.popleft()
            visited_positions.add(current_pos)

            r, c = current_pos

            # Build a list of the moves we can make
            allowed_dirs = [
                d
                for d in range(0, 4)  # 4 moves possible
                if not chess_board[r, c, d] and 
                not adv_pos == (r + self.moves[d][0], c + self.moves[d][1]) and 
                (r + self.moves[d][0], c + self.moves[d][1]) not in visited_positions
            ]

            if steps_remaining == 0 or len(allowed_dirs) == 0:
                # Possibilities, any direction such that chess_board is False
                allowed_barriers = [i for i in range(0, 4) if not chess_board[r, c, i]]

                for barrier_dir in allowed_barriers:
                    potential_move = (current_pos, barrier_dir)
                    if potential_move not in final_positions:
                        final_positions.append(potential_move)
                continue

            for allowed in allowed_dirs:
                m_r, m_c = self.moves[allowed]
                my_new_pos = (r + m_r, c + m_c)

                # Add new position to the queue instead of a recursive call
                if my_new_pos not in visited_positions:
                    queue.append((my_new_pos, steps_remaining - 1))


    def get_moves(self, chess_board, my_pos, adv_pos, max_step, all_moves):
        if self.early_game_max_step:
            self.generate_max_moves_bfs(chess_board, my_pos, adv_pos, max_step, all_moves)
        else:
            self.generate_all_moves_bfs(chess_board, my_pos, adv_pos, max_step, all_moves)

    
    def alpha_beta(self, chess_board, my_pos, adv_pos, depth):  
        
        moves = []

        self.get_moves(chess_board, my_pos, adv_pos, self.max_step, moves)

        best_move = None
        best_score = -sys.maxsize

        for move in moves:
            new_my_pos, new_dir = move
            x, y = new_my_pos

            chess_board[x, y, new_dir] = True
            score = self.minValue(chess_board, new_my_pos, adv_pos, depth - 1, -sys.maxsize, sys.maxsize)
            chess_board[x, y, new_dir] = False

            if score > best_score:
                best_move = move
                best_score = score
    
        return best_move
    
    
    def minValue(self, chess_board, my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(chess_board, my_pos, adv_pos)
        
        moves = []

        self.get_moves(chess_board, my_pos, adv_pos, self.max_step, moves)

        for move in moves:
            new_my_pos, new_dir = move
            x, y = new_my_pos

            chess_board[x, y, new_dir] = True
            alpha = max(alpha,self.maxValue(chess_board, new_my_pos, adv_pos, depth - 1, alpha, beta))
            chess_board[x, y, new_dir] = False

            if beta <= alpha:
                return beta

        return alpha

    def maxValue(self, chess_board, my_pos, adv_pos, depth, alpha, beta):

        if depth == 0:
            return self.evaluate_position(chess_board, my_pos, adv_pos)
        
        moves = []

        self.get_moves(chess_board, my_pos, adv_pos, self.max_step, moves)

        for move in moves:
            new_my_pos, new_dir = move
            x, y = new_my_pos

            chess_board[x, y, new_dir] = True
            min(beta,self.minValue(chess_board, new_my_pos, adv_pos, True, depth - 1, alpha, beta))
            chess_board[x, y, new_dir] = False

            if beta <= alpha:
                return alpha
            
        return beta

    def evaluate_position(self, chess_board, my_pos, adv_pos):
        my_x, my_y = my_pos
        adv_x, adv_y = adv_pos
        
        moves = []

        self.generate_all_moves_bfs(chess_board, my_pos, adv_pos, self.max_step, moves)
        my_moves = len(moves)
      

        count_walls = 0
        for wall in chess_board[my_x, my_y]:
            if wall:
                count_walls += 1
        
        point_modifier = 1
        if count_walls == 3:
            point_modifier = -1000
            
        distance = ((my_x-adv_x)**2 + (my_y-adv_y)**2)**(1/2)
        
        return (my_moves + distance)*point_modifier
    
