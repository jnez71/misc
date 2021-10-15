#!/usr/bin/env python3
"""
The 8-Queens Problem as dynamic programming.
https://en.wikipedia.org/wiki/Eight_queens_puzzle

"""

class QueenSolver:
    def __init__(self, numqueens, boardsize):
        # Cast and validate
        self.numqueens = int(numqueens)
        self.boardsize = int(boardsize)
        assert self.numqueens <= self.boardsize
        # Initialize
        self._horizon = int(self.numqueens)
        self._actions = tuple((i,j) for i in range(self.boardsize) for j in range(self.boardsize))
        self._values = dict()
        self._policy = dict()
        self._State = frozenset
        # Solve
        trajectory = [self._State()]
        self._evaluate(trajectory[0], 0)
        for t in range(self._horizon):
            trajectory.append(self._transition(trajectory[t], self._policy[trajectory[t]]))
        self.solution = trajectory[-1]
        # Visualize
        display = "SOLUTION:\n"
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                display += " Q" if (i,j) in self.solution else " -"
            display += '\n'
        print(display)

    def _evaluate(self, state, time):
        # Boundary conditions
        if self._terminal(state, time):
            return 0.0
        # Memoized regions of the value function so far
        if state in self._values:
            return self._values[state]
        # Dynamic programming principle
        self._values[state] = -1e99
        for action in self._actions:
            value = self._reward(state, action) + self._evaluate(self._transition(state, action), time+1)
            if value >= self._values[state]:
                self._values[state] = value
                self._policy[state] = action
        return self._values[state]

    def _terminal(self, state, time):
        # Are no more queens left?
        if time > self._horizon:
            return True
        # Is the state illegal?
        if state:
            # Have queens merged into the same space?
            if len(state) != time:
                return True
            # Are queens threatening each other?
            rows, cols, ldiags, rdiags = zip(*((q[0], q[1], q[1]-q[0], self.boardsize-(q[0]+q[1])-1) for q in state))
            for i in range(0, len(state)):
                for j in range(i+1, len(state)):
                    if (rows[i] == rows[j]) or (cols[i] == cols[j]) or (ldiags[i] == ldiags[j]) or (rdiags[i] == rdiags[j]):
                        return True
        return False

    def _reward(self, state, action):
        # Same for any placement because illegal placements are (more efficiently) handled by the boundary conditions
        return 1.0

    def _transition(self, state, action):
        # Add the placed queen to the set of already-placed queens
        return self._State((*state, action))


n = 8
qs = QueenSolver(numqueens=n, boardsize=n)
