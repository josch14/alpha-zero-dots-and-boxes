import copy
from typing import Tuple
import numpy as np
import random

# local import
from .player import AIPlayer
from ..game import DotsAndBoxesGame

inf = float("inf")


class AlphaBetaPlayer(AIPlayer):

    def __init__(self, depth: int):
        super().__init__(f"AlphaBetaPlayer(Depth={depth})")
        self.depth = depth

    def determine_move(self, s: DotsAndBoxesGame) -> int:

        # let the first four moves (i.e., the first two moves for a single AlphaBetaPlayer) be random
        # in order to drastically reduce computation time (not for games with size=2)
        if s.SIZE >= 3:
            l = s.l
            if np.count_nonzero(l) < 4:
                # 1) first check whether there already is a box with three lines (simple misplay by opponent)
                for box in np.ndindex(s.b.shape):
                    lines = s.get_lines_of_box(box)
                    if len([line for line in lines if l[line] != 0]) == 3:
                        # there has to be one line which is not drawn yet
                        move = [line for line in lines if l[line] == 0][0]
                        return move

                # 2) moves may only be selected when, after drawing, each box contains a maximum of two drawn lines
                valid_moves = s.get_valid_moves()
                random.shuffle(valid_moves)
                while True:
                    move = valid_moves.pop(0)
                    execute_move = True
                    for box in s.get_boxes_of_line(move):
                        # box should not already have two drawn lines
                        lines = s.get_lines_of_box(box)
                        if len([line for line in lines if l[line] != 0]) == 2:
                            execute_move = False
                    if execute_move:
                        return move


        move, _ = AlphaBetaPlayer.alpha_beta_search(
            s_node=copy.deepcopy(s),
            a_latest=None,
            depth=self.depth,
            alpha=-inf,
            beta=inf,
            maximize=True
        )
        return move


    @staticmethod
    def alpha_beta_search(s_node: DotsAndBoxesGame,  # current node
                          a_latest: int,
                          depth: int,
                          alpha: float,
                          beta: float,
                          maximize: bool) -> Tuple[int, float]:

        valid_moves = s_node.get_valid_moves()
        random.shuffle(valid_moves)  # adds randomness in move selection when multiple moves achieve equal value

        if len(valid_moves) == 0 or depth == 0 or not s_node.is_running():
            # heuristic evaluation
            # if maximize == True, then we know that we (the player for which the
            # search is executed) are the active player. This may be player 1 or player -1
            player = s_node.current_player if maximize else (-1) * s_node.current_player

            player_boxes = (s_node.b == player).sum()
            opponent_boxes = (s_node.b == (-1) * player).sum()

            if not s_node.is_running():
                # game is finished before no valid moves are left
                if player_boxes > opponent_boxes:
                    return a_latest, 10000  # win -> maximum value
                elif player_boxes == opponent_boxes:
                    return a_latest, 0
                else:
                    return a_latest, -10000  # loose -> minimum value

            else:
                return a_latest, player_boxes - opponent_boxes

        if maximize:
            a_best = None
            v_best = -inf

            for a in valid_moves:
                s_child = copy.deepcopy(s_node)
                s_child.execute_move(a)

                player_switched = (s_node.current_player != s_child.current_player)

                _, v_child = AlphaBetaPlayer.alpha_beta_search(
                    s_node=s_child,
                    a_latest=a,
                    depth=depth-1,
                    alpha=alpha,
                    beta=beta,
                    maximize=(not player_switched)
                )

                if v_child > v_best:
                    a_best = a
                    v_best = v_child

                if v_best > beta:
                    break

                alpha = max(alpha, v_best)

            return a_best, v_best

        else:
            a_best = None
            v_best = inf

            for a in valid_moves:
                s_child = copy.deepcopy(s_node)
                s_child.execute_move(a)

                player_switched = (s_node.current_player != s_child.current_player)

                _, v_child = AlphaBetaPlayer.alpha_beta_search(
                    s_node=s_child,
                    a_latest=a,
                    depth=depth-1,
                    alpha=alpha,
                    beta=beta,
                    maximize=player_switched
                )

                if v_child < v_best:
                    a_best = a
                    v_best = v_child

                if v_best < alpha:
                    break

                beta = min(beta, v_best)

            return a_best, v_best
