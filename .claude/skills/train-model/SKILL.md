---
name: train-model
description: Học hàm đánh giá của bot Wrongway qua tự đấu - học tăng cường TD-leaf(lambda) cho các bảng tra (training/td_learn.py, đang dùng) hoặc thuật toán tiến hóa cho bản tuyến tính cũ (training/evolve.py) - rồi ghi vào khối WEIGHTS trong MSSV.py. Dùng khi người dùng muốn train, train tiếp, thêm/bớt đặc trưng đánh giá, hoặc xem tiến độ train.
---

# Train hàm đánh giá (TD-leaf + alpha-beta; tiến hóa cho bản cũ)

## Kiến trúc
- `submissions/MSSV.py` (thư viện chuẩn):
  - Tự mô phỏng luật (`edge_open`, `legal_moves`, `wall_fits`, `distance`...) — phải khớp `environment.py`; `tests/test_bot.py::RulesMatchArena` kiểm tra điều này.
  - `Search.negamax`: alpha-beta, đào sâu dần tới `MAX_DEPTH` trong `TIME_BUDGET`. Ứng viên: mọi nước đi + `WALL_CANDIDATES` tường tốt nhất chắn đường ngắn nhất của đối thủ.
  - `evaluate(p, dist, wl, vuln)`: tổng các ô **tra bảng** ở khối `WEIGHTS BEGIN/END`: `tempo + safe_race*safe + side_value(mình) - side_value(đối thủ) + vuln[w_mình]*vuln_đối_thủ - vuln[w_đối_thủ]*vuln_mình`, với `side_value(X) = dist[d_X] + walls[w_X][d_Y] + lead[w_X][d_Y - d_X]` (428 tham số).
- `training/td_learn.py` (cần numpy; **đang dùng**, thử nghiệm #36–#37): TD-leaf(λ). Bot tự đấu bằng chính `negamax` của nó ở độ sâu 5–6; mỗi lượt lấy thế cờ lá của biến chính (từ bảng chuyển vị); v = sigmoid(±E/K); sai số TD d_t = v_{t+1} − v_t; mỗi ô có mặt ở lá được chỉnh theo e_t = Σ λ^k d_{t+k}. Mỗi vòng ghi bảng vào file `--bot` và `training/checkpoints/<tag>/iter_XXX.json`.
- `training/evolve.py` (**chỉ cho hàm đánh giá tuyến tính cũ**, `--bot training/opponents/v7_linear.py`; từ chối file có trọng số dạng bảng): (1+λ)-ES. Đột biến đấu với nhà vô địch (đổi bên, mở đầu ngẫu nhiên) + bot mẫu; thắng >= gate và qua vòng xác nhận → ghi vào MSSV.py. Lịch sử: `training/checkpoints/evolve_history.jsonl`; bản trước mỗi lần thay: `MSSV_before_gen_XXX.py`.

## Lệnh (từ thư mục gốc)
```
# TD: luyện trên bản sao dev.py, KHÔNG luyện thẳng trên MSSV.py (mỗi vòng ghi đè bảng)
cp wrongway-student/wrongway/submissions/MSSV.py wrongway-student/wrongway/submissions/dev.py
python -u training/td_learn.py --bot wrongway-student/wrongway/submissions/dev.py --iterations 20 --games 200 \n    --depth 5,6 --eta 0.3 --eval-every 4 --eval-pairs 30 --workers 10 --tag td3 > logs/td3.log 2>&1   # ~5 phút/vòng
# Tiến hóa (bản tuyến tính cũ):
python -u training/evolve.py --generations 25 --workers 10 > logs/evolve.log 2>&1   # chạy nền
```
Tham số chính: `--lam` (số đột biến), `--pairs` (cặp ván vs vô địch), `--gate` (0.55), `--budget`/`--depth` (thời gian/độ sâu khi train, nhỏ hơn khi nộp để nhanh), `--sigma`.
Chạy lại sẽ tiếp tục từ trọng số hiện có trong MSSV.py.

Lưu ý khi dùng TD (bài học #36–#37):
- **Độ sâu tự đấu phải gần độ sâu thi đấu (5–6).** Luyện ở độ sâu 2–3 cho bảng thắng bảng đầu 0.68 ở độ sâu 3–4 nhưng thua bản gốc 0.34 ở thời gian thật.
- Cột `vs bảng đầu` trong log rất nhiễu (60 ván, ±0.065) và ở độ sâu cố định: chỉ để phát hiện hỏng nặng. Nhận hay bỏ phải đo bằng `gauntlet.py` ở tốc độ Colab (`TIME_BUDGET = 0.275`), cùng seed với bản gốc, ≥ 200 ván trước AI Hard và ≥ 120 ván trước Challenger.
- `--only pos,near_h` chỉ học các bảng được nêu (để đo riêng tác dụng của bảng mới); `--avg-from N` ghi thêm `avg.json` = trung bình bảng từ vòng N (bảng của một vòng dao động, bản trung bình ổn định hơn).
- Bảng "thế đứng" (ô quân đứng + tường quanh quân, #38) đã thử và bỏ; bot mẫu: `training/opponents/td3_stance.py` (trình luyện tự nhận ra khi `WEIGHTS` có khóa `pos`).
- Thêm bảng mới: thêm khóa vào `WEIGHTS`, dùng nó trong `evaluate`/`leaf` của bot **và** trong `Layout.features` của td_learn.py (lệnh chạy tự kiểm tra hai bên khớp nhau); khởi đầu bằng 0 để không đổi hành vi.

## Thêm một đặc trưng mới (bản tuyến tính + tiến hóa)
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

Lưu ý vấn đáp: người dùng cần giải thích được negamax/alpha-beta, đào sâu dần, ý nghĩa từng bảng (`dist`, `walls`, `lead`, `vuln`), TD-learning là gì (dự đoán ở lượt trước được kéo về phía dự đoán ở lượt sau / kết quả ván; λ; vì sao lấy thế cờ lá của biến chính), vì sao bảng khởi đầu là công thức tuyến tính, vì sao tiến hóa không bị tương quan giả (mỗi bộ trọng số được chấm bằng trận thật), và cơ chế gate/xác nhận chống nhiễu.
