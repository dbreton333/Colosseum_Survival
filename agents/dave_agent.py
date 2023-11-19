# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time


@register_agent("dave_agent")
class daveAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(daveAgent, self).__init__()
        self.name = "dave_agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.max_step = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()

        if self.max_step is None:
            self.max_step = max_step

        best_move = self.alpha_beta(chess_board, my_pos, adv_pos, 2)

        new_pos, new_dir = best_move

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return new_pos, new_dir
    

    def generate_all_moves_dfs(self, chess_board, my_pos, adv_pos, max_step, all_moves, visited_positions):

        # Moves (Up, Right, Down, Left)
        if max_step <= 0 or my_pos in visited_positions:
            # Termination condition: If max_step is non-positive or already visited, stop recursion
            return all_moves

        visited_positions.add(my_pos)  # Mark the current position as visited

        r, c = my_pos
        # Build a list of the moves we can make
        allowed_dirs = [
            d
            for d in range(0, 4)  # 4 moves possible
            if not chess_board[r, c, d] and  # chess_board True means wall
            not adv_pos == (r + self.moves[d][0], c + self.moves[d][1])  # cannot move through Adversary
        ]

        for allowed in allowed_dirs:
            # This is how to update a row, col by the entries in moves
            # to be consistent with game logic
            m_r, m_c = self.moves[allowed]
            my_new_pos = (r + m_r, c + m_c)

            # Possibilities, any direction such that chess_board is False
            allowed_barriers = [i for i in range(0, 4) if not chess_board[r + m_r, c + m_c, i]]

            for barrier_dir in allowed_barriers:
                potential_move = (my_new_pos, barrier_dir)
                all_moves.append(potential_move)

            all_moves = self.generate_all_moves_dfs(chess_board, my_new_pos, adv_pos, max_step - 1, all_moves, visited_positions)

        return all_moves
    
    def alpha_beta(self, chess_board, my_pos, adv_pos, depth):  
            
            moves = []
            visited_positions = set()
    
            self.generate_all_moves_dfs(chess_board, my_pos, adv_pos, self.max_step, moves, visited_positions)
    
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
        visited_positions = set()

        self.generate_all_moves_dfs(chess_board, my_pos, adv_pos, self.max_step, moves, visited_positions)

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
        visited_positions = set()

        self.generate_all_moves_dfs(chess_board, my_pos, adv_pos, self.max_step, moves, visited_positions)

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
      # Heuristic function
        # Right now heuristic is number of moves available to us versus opponent
        # my_moves = len(self.generate_all_moves_prune(chess_board, my_pos, adv_pos, max_step, None))
        # their_moves = len(self.generate_all_moves_prune(chess_board, adv_pos, my_pos, max_step, None))
        # return my_moves - their_moves
        
        # This heuristic prioritizes how many moves we have and our distance from the opponent
        # It also gives a big negative value if we have three walls surrounding us based on the move, want to avoid those
        my_x, my_y = my_pos
        adv_x, adv_y = adv_pos
        
        moves = []
        visited_positions = set()

        my_moves = len(self.generate_all_moves_dfs(chess_board, my_pos, adv_pos, self.max_step, moves, visited_positions))
      

        count_walls = 0
        for wall in chess_board[my_x, my_y]:
            if wall:
                count_walls += 1
        
        point_modifier = 1
        if count_walls == 3:
            point_modifier = -1000
            
        distance = ((my_x-adv_x)**2 + (my_y-adv_y)**2)**(1/2)
        
        return (my_moves + distance)*point_modifier
    
