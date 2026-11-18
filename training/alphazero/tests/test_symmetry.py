from training.alphazero.engine import E
from training.alphazero.symmetry import (FEATURE_MIRROR, N_ACTIONS, POLICY_MIRROR,
                                         canonical_action, canonical_policy, rotate_for)
from training.alphazero.tests.helpers import random_states


def test_feature_mirror_matches_encode_mirror():
    for state in list(random_states(n_games=4, seed=5))[:120]:
        plain = E.encode(state)
        assert [plain[i] for i in FEATURE_MIRROR] == E.encode(state, mirror=True)


def test_policy_mirror_is_an_involution():
    assert sorted(POLICY_MIRROR) == list(range(N_ACTIONS))
    assert all(POLICY_MIRROR[POLICY_MIRROR[a]] == a for a in range(N_ACTIONS))


def test_canonical_policy_puts_mass_on_canonical_index():
    state = E.State(turn=1)                  # player 1 -> rotated
    assert rotate_for(state)
    pi = canonical_policy(state, {4: 3, E.PASS: 1})
    assert abs(sum(pi) - 1.0) < 1e-9
    assert pi[canonical_action(state, 4)] == 0.75
    assert pi[E.PASS] == 0.25


def test_mirrored_policy_lines_up_with_mirrored_encoding():
    state = E.State(turn=1)
    for action in E.legal_actions(state):
        j = canonical_action(state, action)
        assert POLICY_MIRROR[j] == E.transform_action(action, True, True)
