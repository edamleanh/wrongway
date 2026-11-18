from training.alphazero.games import (_init_worker, bootstrap_game, eval_game, game_summary,
                                      run_parallel, selfplay_game)
from training.alphazero.net import AZNet, save_checkpoint

SP_PARAMS = {"c_puct": 1.5, "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25, "opening_plies": 2,
             "ply_cap": 12, "simulations": 4, "temp_moves": 4}


def check_records(records):
    assert records
    for features, pi, z in records:
        assert len(features) == 651 and len(pi) == 210
        assert abs(sum(pi) - 1.0) < 1e-6
        assert z in (-1.0, 0.0, 1.0)


def test_bootstrap_game():
    records, stats = bootstrap_game((1, 0.01, 8, 2, 16))
    check_records(records)
    assert stats["plies"] <= 16


def test_selfplay_and_eval_games(tmp_path):
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, AZNet(), 0)
    _init_worker({"model": str(ckpt), "new": str(ckpt), "old": str(ckpt)})
    records, stats = selfplay_game((3, SP_PARAMS))
    check_records(records)
    assert eval_game((5, 0, 0.02, 8, 2, 12, "old")) in (0.0, 0.5, 1.0)
    assert eval_game((5, 1, 0.02, 8, 2, 12, "base")) in (0.0, 0.5, 1.0)   # vs the no-network engine


def test_run_parallel_uses_processes():
    jobs = [(seed, 0.01, 8, 2, 10) for seed in range(2)]
    results = list(run_parallel(bootstrap_game, jobs, workers=2))
    assert len(results) == 2


def test_game_summary():
    s = game_summary([{"plies": 10, "winner": 0}, {"plies": 20, "winner": None}])
    assert s == {"games": 2, "avg_plies": 15.0, "p0_win": 0.5, "p1_win": 0.0, "draw": 0.5}
