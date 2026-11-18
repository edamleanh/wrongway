"""Canonical action frame and left/right mirror symmetry.

encode() rotates the board so the side to move always heads for row 0. Network policy
outputs live in that frame: output j <-> real action transform_action(j, rotate).
Mirroring columns is an exact symmetry of Wrongway, so every sample can be flipped for
free data augmentation: x_mirror = x[FEATURE_MIRROR], pi_mirror = pi[POLICY_MIRROR]."""
from .engine import E

N_FEATURES = 651
N_ACTIONS = E.ACTION_COUNT           # 210 = 81 moves + 64 H walls + 64 V walls + pass


def rotate_for(state):
    return (state.goal1 if state.turn % 2 else state.goal0) == 8


def canonical_action(state, action):
    return E.transform_action(action, rotate_for(state))


def canonical_policy(state, visits):
    """{real action: count} -> length-210 distribution in the canonical frame."""
    pi = [0.0] * N_ACTIONS
    total = float(sum(visits.values()))
    if total > 0:
        for action, count in visits.items():
            pi[canonical_action(state, action)] = count / total
    return pi


def _feature_mirror():
    perm = list(range(N_FEATURES))
    for offset in (0, 81, 324, 405, 486, 567):       # 9x9 cell planes: pawns, distances, goal rows
        for cell in range(81):
            perm[offset + cell] = offset + E.transform_cell(cell, False, True)
    for offset in (162, 243):                         # wall planes: slot (row, col) stored at row*9+col
        for row in range(8):
            for col in range(8):
                perm[offset + row * 9 + col] = offset + row * 9 + (7 - col)
    return perm                                       # 648..650 (scalars) stay put


FEATURE_MIRROR = _feature_mirror()
POLICY_MIRROR = [E.transform_action(a, False, True) for a in range(N_ACTIONS)]
