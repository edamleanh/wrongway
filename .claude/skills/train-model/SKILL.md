---
name: train-model
description: Học trọng số hàm đánh giá của bot Wrongway bằng thuật toán tiến hóa qua tự đấu (training/evolve.py), rồi ghi vào khối WEIGHTS trong MSSV.py. Dùng khi người dùng muốn train, train tiếp, thêm/bớt đặc trưng đánh giá, hoặc xem tiến độ train.
---

# Train trọng số (tiến hóa + alpha-beta)

## Kiến trúc
- `submissions/MSSV.py` (thư viện chuẩn):
  - Tự mô phỏng luật (`edge_open`, `legal_moves`, `wall_fits`, `distance`...) — phải khớp `environment.py`; `tests/test_bot.py::RulesMatchArena` kiểm tra điều này.
  - `Search.negamax`: alpha-beta, đào sâu dần tới `MAX_DEPTH` trong `TIME_BUDGET`. Ứng viên: mọi nước đi + `WALL_CANDIDATES` tường tốt nhất chắn đường ngắn nhất của đối thủ.
  - `evaluate(p, dist, wl)`: tổng có trọng số các đặc trưng; trọng số ở khối `WEIGHTS BEGIN/END` (dict).
- `training/evolve.py`: (1+λ)-ES. Đột biến đấu với nhà vô địch (đổi bên, mở đầu ngẫu nhiên) + bot mẫu; thắng >= gate và qua vòng xác nhận → ghi vào MSSV.py. Lịch sử: `training/checkpoints/evolve_history.jsonl`; bản trước mỗi lần thay: `MSSV_before_gen_XXX.py`.

## Lệnh (từ thư mục gốc)
```
python -u training/evolve.py --generations 25 --workers 10 > logs/evolve.log 2>&1   # chạy nền
```
Tham số chính: `--lam` (số đột biến), `--pairs` (cặp ván vs vô địch), `--gate` (0.55), `--budget`/`--depth` (thời gian/độ sâu khi train, nhỏ hơn khi nộp để nhanh), `--sigma`.
Chạy lại sẽ tiếp tục từ trọng số hiện có trong MSSV.py.

## Thêm một đặc trưng mới
1. Tính nó trong `evaluate()` và thêm khóa vào `WEIGHTS` (giá trị khởi đầu 0.0 → không đổi hành vi).
2. Thêm khóa với độ lớn đột biến vào `SCALES` trong evolve.py.
3. Chạy test, rồi evolve.

## Đo sức mạnh và thử nghiệm engine
- `python training/gauntlet.py MSSV training/opponents/v2_trained.py wall_builder --pairs 20 --workers 6` — 40 ván/đối thủ; chênh < ~0.1 là nhiễu, đo lại với `--seed` khác trước khi kết luận.
- Đọc `training/EXPERIMENTS.md` trước (đã thử quiescence, phạt lặp, WALL_CANDIDATES, TIME_BUDGET) và ghi kết quả mới vào đó.
- Bản vô địch mới cần lưu vào `training/opponents/` để lần tiến hóa sau dùng làm đối thủ chuẩn.

## Sau khi train
1. `python -m unittest discover tests`
2. `python tools/tournament.py MSSV --vs-all --seeds 3` (dùng TIME_BUDGET thật)
3. `python tools/check_submission.py wrongway-student/wrongway/submissions/MSSV.py`
4. Tệ hơn → khôi phục từ `training/checkpoints/MSSV_before_gen_XXX.py`.

Lưu ý vấn đáp: người dùng cần giải thích được negamax/alpha-beta, đào sâu dần, ý nghĩa từng đặc trưng, vì sao tiến hóa không bị tương quan giả (mỗi bộ trọng số được chấm bằng trận thật), và cơ chế gate/xác nhận chống nhiễu.
