#!/usr/bin/env python3
"""
Coverage path planning as dynamic programming.

"""

from typing import Tuple, NamedTuple, FrozenSet
from math import inf


class State(NamedTuple):
    t: int  # time remaining
    p: Tuple[int]  # grid position
    c: FrozenSet[Tuple[int]]  # covered cells


actions = frozenset((
    ( 1, 0),  # right
    (-1, 0),  # left
    ( 0, 1),  # up
    ( 0,-1),  # down
))


obstacles = frozenset((
    (2,4),
    (2,5),
    (3,3),
    (6,3),
    (6,4),
    (9,6),
    (9,7),
))


interests = frozenset((
    (3,5), (4,5), (5,5), (6,5), (7,5),
    (3,6), (4,6), (5,6), (6,6), (7,6),
    (3,7), (4,7), (5,7), (6,7), (7,7),
))


start  = (2,1)
assert start not in obstacles
assert start not in interests


finish = (6,0)
assert finish not in obstacles
assert finish not in interests


horizon = 16  # seconds
def duration(s, a):
    # # Perhaps going leftward is slower (like upstream)
    # if a[0] < 0:
    #     return 2
    # Nominal transitions take 1 unit of time
    return 1


def transition(s, a):
    # Out of time
    if s.t <= 0:
        return s
    # Candidate movement
    p = (s.p[0]+a[0], s.p[1]+a[1])
    t = s.t - duration(s, a)
    # Check for termination
    if p == finish:
        return State(0, p, s.c)
    # Check for collision
    if p in obstacles:
        return State(t, s.p, s.c)
    # Check for coverage
    if p in interests:
        return State(t, p, s.c.union((p,)))
    # Typical result
    return State(t, p, s.c)


def reward(s, a):
    # Slight penalty for dilly-dallying
    if s.p != finish:
        return -1e-9
    # Main reward for coverage is handled at boundary condition
    return 0.0


values = {}
policy = {}
def evaluate(s):
    global values, policy
    # Check cache
    if s in values:
        return values[s]
    # Boundary condition
    if s.t <= 0:
        # Count coverage if at finish, unacceptable otherwise
        values[s] = float(len(s.c)) if (s.p == finish) else -inf
        return values[s]
    # Admissible heuristic (check if you can't make it back in time)
    if (abs(finish[0]-s.p[0]) + abs(finish[1]-s.p[1])) > s.t:
        values[s] = -inf
        return values[s]
    # Dynamic programming principle
    values[s] = -inf
    for a in actions:
        v = reward(s, a) + evaluate(transition(s, a))
        if v >= values[s]:
            values[s] = v
            policy[s] = a
    return values[s]


state = State(horizon, start, frozenset())
score = evaluate(state)


path = [state.p]
while state.t > 0:
    state = transition(state, policy[state])
    path.append(state.p)


n = (10, 8)
from matplotlib import pyplot, patches
figure, axes = pyplot.subplots(1, 1)
axes.set_title(f"Optimal Path for T={horizon}", fontsize=18)
axes.set_xlim([-0.5, n[0]-0.5])
axes.set_ylim([-0.5, n[1]-0.5])
axes.plot(*zip(*path), color='purple', zorder=2)
for x in range(n[0]):
    for y in range(n[1]):
        color = 'white'
        if (x,y) in (start, finish):
            color = 'gray'
        elif (x,y) in obstacles:
            color = 'red'
        elif (x,y) in interests:
            color = 'green'
        axes.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=color, edgecolor="black", linewidth=0.2, zorder=1))
pyplot.show()
