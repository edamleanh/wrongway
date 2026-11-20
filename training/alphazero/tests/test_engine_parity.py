"""The forked agent's rules must match the instructor Arena exactly."""
import random

from training.alphazero.engine import E
from training.alphazero.tests.helpers import Environment, arena_actions, arena_envs


def test_legal_actions_match_arena():
    checked = 0
    for env in arena_envs():
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        assert set(E.legal_actions(state)) == arena_actions(obs)
        checked += 1
    assert checked > 300


def test_distances_match_arena():
    for env in arena_envs(n_games=10):
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        ours = E.distances(state.h, state.v, obs["you"]["goal_row"])
        assert list(ours) == [d for row in obs["distances"] for d in row]


def test_decoder_handles_player1_starting():
    for env in arena_envs(n_games=5, starting_player=1):
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        me = state.p1 if state.turn % 2 else state.p0
        assert [me // 9, me % 9] == obs["you"]["position"]
        assert set(E.legal_actions(state)) == arena_actions(obs)


def test_play_matches_arena_step():
    rng = random.Random(3)
    for _ in range(10):
        env = Environment(max_turns=120)
        state = E.state_from_observation(env.get_observation(env.current_player))
        while not env.done:
            action = rng.choice(sorted(E.legal_actions(state)))
            env.step(E.action_dict(action))
            state = E.play(state, action, validate=False)
            if env.done:
                assert E.outcome(state) == (2 if env.winner is None else env.winner)
                break
            assert state == E.state_from_observation(env.get_observation(env.current_player))
