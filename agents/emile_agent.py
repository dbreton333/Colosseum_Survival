# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import sys
import time


@register_agent("emile_agent")
class EmileAgent(Agent):

    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(EmileAgent, self).__init__()
        self.name = "EmileAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.turn = 0
        self.moves_checked = 0
        self.turn_time = 0
        self.max_depth = 0
        self.max_time = 0
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
        if(self.turn == 0):
            self.max_depth = 3
            self.max_time = 2
        else:
            if chess_board.shape[0] > 7:
                self.max_depth = 1
            else:
                self.max_depth = 2
            self.max_time = 1.95
        self.turn_time = time.time()
        start_time = time.time()
        self.turn += 1
        self.moves_checked = 0
        move = self.search(chess_board, my_pos, adv_pos, max_step)
        end_time = time.time()
        #print("Checked " + str(self.moves_checked) + " moves in " + str(end_time-start_time) + " seconds")
        return move[0], self.dir_map[move[1]]

    def minimaxValue(self, chess_board, p1_pos, p2_pos, max_step, depth, isMax, alpha, beta):
        self.moves_checked += 1
        endgame = self.check_endgame(chess_board, p1_pos, p2_pos)
        my_pos = p1_pos if isMax else p2_pos
        adv_pos = p2_pos if isMax else p1_pos
        if(endgame[0]):
            if(isMax):
                if(endgame[1] > endgame[2]):
                    return 100-depth
                elif(endgame[1] < endgame[2]):
                    return -100+depth
                else:
                    return 0
            else:
                if(endgame[1] > endgame[2]):
                    return -100+depth
                elif(endgame[1] < endgame[2]):
                    return 100-depth
                else:
                    return 0

        tiles = self.get_available_tiles(chess_board, p1_pos, p2_pos, max_step)
        if not tiles:
            if(isMax):
                return -100+depth
            else:
                return 100-depth

        if(depth > self.max_depth):
            score = 0
            if(isMax):
                my_tiles = tiles
                adv_tiles = self.get_available_tiles(chess_board, adv_pos, my_pos, max_step)
                score += 2*len(my_tiles) - 2*len(adv_tiles)
            else:
                adv_tiles = tiles
                my_tiles = self.get_available_tiles(chess_board, my_pos, adv_pos, max_step)
                score += 2*len(my_tiles) - 2*len(adv_tiles)

            my_tiles = self.get_available_tiles(chess_board, my_pos, adv_pos, max_step*2)
            adv_tiles = self.get_available_tiles(chess_board, adv_pos, my_pos, max_step*2)
            score += len(my_tiles) - len(adv_tiles)

            return score

        dirs = self.dir_map.keys()
        # if(isMax):
        #     best_value = -sys.maxsize
        # else:
        #     best_value = sys.maxsize
        for tile in tiles:
            for dir in dirs:
                if(chess_board[(tile[0])[0], (tile[0])[1], self.dir_map[dir]]):
                    continue
                p1_pos = tile[0]
                new_chess_board = deepcopy(chess_board)
                new_chess_board[(tile[0])[0], (tile[0])[1], self.dir_map[dir]] = True
                res = self.minimaxValue(new_chess_board, p2_pos, p1_pos, max_step, depth+1, not isMax, alpha, beta)
                if(isMax):
                    if(res > alpha):
                        alpha = res
                    if(alpha >= beta):
                        return alpha
                else:
                    if(res < beta):
                        beta = res
                    if(alpha >= beta):
                        return beta
                if(time.time() - self.turn_time > self.max_time):
                    if(isMax):
                        return alpha
                    else:
                        return beta
        if(isMax):
            return alpha
        else:
            return beta


    # Search for the best move
    # Use minimax with alpha-beta pruning
    def search(self, chess_board, p1_pos, p2_pos, max_step):
        moves = []
        alpha = -sys.maxsize
        beta = sys.maxsize
        dirs = self.dir_map.keys()
        #best_value = -sys.maxsize
        best_move = p1_pos
        tiles = self.get_available_tiles(chess_board, p1_pos, p2_pos, max_step)
        for tile in tiles:
            for dir in dirs:
                if(chess_board[(tile[0])[0], (tile[0])[1], self.dir_map[dir]]):
                    continue
                p1_pos = tile[0]
                new_chess_board = deepcopy(chess_board)
                new_chess_board[(tile[0])[0], (tile[0])[1], self.dir_map[dir]] = True
                res = self.minimaxValue(new_chess_board, p2_pos, p1_pos, max_step, 1, False, alpha, beta)
                moves.append((tile, dir, res))

                if(res > alpha):
                    alpha = res
                    best_move = (tile[0], dir)
                    if(alpha >= beta):
                        break

        return best_move

        


    def get_available_tiles(self, chess_board, my_pos, adv_pos, max_step):
        tiles = [(my_pos,0)]
        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]) or tuple(next_pos) in visited:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
                tiles.append((next_pos, cur_step))
                #tiles.insert(0,(next_pos, cur_step))
        
        return tiles

    def check_endgame(self, chess_board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """

        board_size = chess_board.shape[0]
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score
