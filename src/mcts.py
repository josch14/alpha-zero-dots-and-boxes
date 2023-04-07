import copy
import math
from random import randint
import numpy as np

# local imports
from .game import DotsAndBoxesGame
from .model.neural_network import AZNeuralNetwork
from .node import AZNode


class MCTS:
    """
    An MCTS is executed, guided by the neural network, in each position s during self-play to determine the next move.

    Attributes
    ----------
    model : AZNeuralNetwork
        neural network for evaluating board positions
    root : AZNode
        node from which the MCTS is executed (with input position s)
    n_simulations : int
        # simulations for each MCTS (only to determine the next move)
    c_puct : float
        constant determining level of exploration (PUCT algorithm in select)
    dirichlet_eps : float
        weight of dirichlet noise for root node of a simulation
    dirichlet_alpha : float
        distribution parameter for dirichlet noise
    """

    def __init__(self, model: AZNeuralNetwork, s: DotsAndBoxesGame, mcts_parameters: dict):

        self.model = model
        self.root = AZNode(
            parent=None,
            a=None,
            s=s
        )

        self.n_simulations = mcts_parameters["n_simulations"]
        self.c_puct = mcts_parameters["c_puct"]
        self.dirichlet_eps = mcts_parameters["dirichlet_eps"]
        self.dirichlet_alpha = mcts_parameters["dirichlet_alpha"]

    def play(self, temp: int) -> [float]:
        """
        (d) Play.
        Provides the core functionality of MCTS: output search probabilities recommending moves to play.

        Parameters
        ----------
        temp : int
            temperature controlling parameter

        Returns
        -------
        probs : [float]
            move probabilities pi(a) ~ N(s,a)^(1/temp)
        """

        s = self.root.s  # position s of root node (more accurate: the game state that contains position s)
        valid_moves = self.root.s.get_valid_moves()

        # perform MCTS simulations
        for i in range(self.n_simulations):
            # dirichlet noise (only for valid moves), added later to the prior probabilities of the root node
            dirichlet_noise = np.zeros((s.N_LINES,), dtype=np.float32)
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

            self.search(self.root, is_root=True, dirichlet_noise=dirichlet_noise)

        # only valid moves may have a visit
        assert set(list(self.root.N.keys())).issubset(set(s.get_valid_moves()))

        # probability vector that is returned should contain value for each line
        # when line is already drawn, probability should be 0
        counts = [self.root.N[a] if a in self.root.N else 0 for a in range(s.N_LINES)]

        if temp == 0:
            # select the move with maximum visit count to give the strongest possible play (return value is one-hot vector)
            probs = [0] * len(counts)
            probs[np.array(counts).argmax()] = 1
            return probs

        # pi(a) ~ N(s,a)^(1/temp) while ensuring a probability distribution
        probs = [n ** (1. / temp) for n in counts]
        probs = [p / float(sum(probs)) for p in probs]
        return probs


    def search(self, node: AZNode, is_root: bool = False, dirichlet_noise: np.ndarray = None) -> float:
        """
        Recursively perform a single simulation within MCTS.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        is_root : bool
            whether the current node is the root node of the search or not (relevant for dirichlet noise)
        dirichlet_noise : bool
            dirichlet noise that is applied only on the prior probabilities of the root node

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """

        if not node.s.is_running():
            # game is finished before reaching a non-visited node
            # return the actual score v for the current player
            # in case of a winner, current_player contains it (when capturing
            # a box, the current player does not switch)
            result = node.s.result

            if node.s.current_player == result:
                v = 1
            elif result == 0:
                v = 0
            else:
                v = -1
            return v

        # reached a leaf (node which was not visited yet)
        if node.P is None:
            v = self.evaluate(node)
            return v

        # node was visited before: continue traversing the tree
        a = self.select(node, is_root, dirichlet_noise)

        if a not in node.N:
            # applying the selected move means approaching a leaf (node which was not visited yet)
            child = self.expand(node, a)
        else:
            child = node.get_child_by_move(a)

        # continue traversing, i.e., call method recursively
        v_child = self.search(child)

        # we now received a score v from the child node, either by ..
        # .. reaching a leaf (v in [-1,1] as calculated by the neural network) or by
        # .. finishing the game (v in {-1, 0, 1})
        v = v_child if node.s.current_player == child.s.current_player else -v_child

        # backup before returning v
        self.backup(node, a, v)

        return v

    def select(self, node: AZNode, is_root: bool, dirichlet_noise: np.ndarray) -> int:
        """
        (a) Select.
        Select the move with maximum action value Q, plus an upper confidence bound U that depends on a stored
        prior probability P and visit count N.

        Parameters
        ----------
        node : AZNode
            (non-leaf) node that corresponds to the MCTS's current position s
        is_root : bool
            whether the current node is the root node of the search or not (relevant for dirichlet noise)
        dirichlet_noise : bool
            dirichlet noise that is applied only on the prior probabilities of the root node

        Returns
        -------
        a_max : int
            move a for which Q(s,a) + U(s,a) is maximized
        """
        assert len(node.s.get_valid_moves()) > 0

        maximum = float('-inf')
        a_max = -1

        N_sum = sum(node.N.values())
        N_sqrt = math.sqrt(N_sum)

        P = node.P if not is_root else \
            (1 - self.dirichlet_eps) * node.P + self.dirichlet_eps * dirichlet_noise
        assert abs(np.sum(P) - 1) < 1e-6, \
            f"is_root: {is_root}, sum of P: {np.sum(node.P)}, sum of P after adding dirichlet noise: {np.sum(P)}"

        for a in node.s.get_valid_moves():
            # each move corresponds to a child node that may or may not have
            # already been visited

            p = P[a]
            if a in node.N:
                q = node.Q[a]
                n = node.N[a]
            else:
                q = 0
                n = 0

            # upper confidence bound U(s, a) ~ P(s, a) / (1 + N(s, a))
            u = self.c_puct * p * N_sqrt / (1 + n)

            # maximize action value Q(s,a) + upper confidence bound U(s,a)
            if q + u > maximum:
                maximum = q + u
                a_max = a

        return a_max


    def expand(self, node: AZNode, a: int):
        """
        (b) Expand (and Evaluate).
        For the input node, create the child node (i.e., we are approaching a leaf) that is reached when executing move a.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        a : int
            move with which the leaf is reached from the current node

        Returns
        -------
        leaf : AZNode
            the created child node/leaf
        """
        s = copy.deepcopy(node.s)
        s.execute_move(a)
        leaf = AZNode(
            parent=node,
            a=a,
            s=s
        )
        return leaf


    def evaluate(self, leaf: AZNode) -> float:
        """
        (b) (Expand and) Evaluate.
        Evaluate the associated position of the leaf node by the neural network
        and store the vector of P values.

        Parameters
        ----------
        leaf : AZNode
            (leaf) node that corresponds to the MCTS's current position s

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """

        # rules are invariant to colour transposition: represent the board from the perspective of the current player
        canonical_lines = leaf.s.get_canonical_lines()
        canonical_boxes = leaf.s.get_canonical_boxes()

        # neural network evaluation is carried out on a reflection or rotation which is selected uniformly
        i = randint(0, 7)
        j = i
        if i == 1:
            j = 3
        elif i == 3:
            j = 1

        lines = DotsAndBoxesGame.get_rotations_and_reflections_lines(canonical_lines)[i]
        boxes = DotsAndBoxesGame.get_rotations_and_reflections_boxes(canonical_boxes)[i]

        # get prediction, and apply same revert rotation that was applied to lines vector before forwarding to neural network
        p, v = self.model.p_v(lines, boxes)
        p = DotsAndBoxesGame.get_rotations_and_reflections_lines(p)[j]

        leaf.P = p
        return v

    def backup(self, node: AZNode, a: int, v: float):
        """
        (c) Backup.
        Update action value Q to track the mean of all evaluations v in the subtree below that node, and visit count N.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        a : int
            move that was selected and executed within the current search
        v : float
            the resulting score for this node for the current simulation
        """

        if a not in node.N:
            # leaf: node was visited for the first time
            node.Q[a] = v
            node.N[a] = 1

        else:
            n = node.N[a]
            q = node.Q[a]
            node.Q[a] = (n * q + v) / (n + 1)  # Q = mean of v
            node.N[a] += 1


    @staticmethod
    def determine_move(model: AZNeuralNetwork, s: DotsAndBoxesGame, mcts_parameters: dict) -> int:

        mcts = MCTS(model, s, mcts_parameters)
        probs = mcts.play(
            temp=0  # select the move with maximum visit count, to give the strongest possible play
        )
        move = np.array(probs).argmax()

        valid_moves = np.where(s.l == 0)[0].tolist()
        assert move in valid_moves, f"move {move} is not a valid move in {valid_moves}"

        return move
