---
name: run-match
description: Chạy một trận Wrongway giữa hai bot (mặc định bot của mình MSSV vs một bot mẫu), có thể render bàn cờ, lưu và xem lại replay. Dùng khi người dùng muốn "chạy thử", "đấu thử", "xem trận", "replay".
---

# Chạy một trận Wrongway

Chạy từ thư mục gốc dự án. Trên Windows dùng `python` (hoặc `py -3`).

1. Xác định hai người chơi. Mặc định: `MSSV` vs `wall_builder`. Bot mẫu có: `greedy`, `random_agent`, `wall_builder`. Tên không cần đuôi `.py`; Arena tìm trong `wrongway-student/wrongway/submissions/`.
2. Chạy:
   ```
   python wrongway-student/wrongway/arena.py -p1 <A> -p2 <B> --seed 0 --log logs/<A>_vs_<B>.json
   ```
   Thêm `--render` nếu người dùng muốn xem từng nước (output dài, chỉ dùng khi được yêu cầu hoặc khi debug một tình huống cụ thể).
3. **Luôn chạy cả hai bên**: đổi `-p1`/`-p2` để bot của mình làm cả P0 (đi trước, ở dưới) và P1.
4. Báo cáo: người thắng, số lượt, và thống kê `timeout/invalid/error` của mỗi bên. Nếu có `error`, đọc trường `error` trong log JSON (`turns[i].error`) để tìm traceback.
5. Xem lại replay: `python wrongway-student/wrongway/replay.py logs/<file>.json --delay 0.3`.

Để phân tích log bằng code: log là JSON với `turns[]` (mỗi phần tử có `turn`, `player`, `action`, `status`, `error`, `elapsed`, `state`), `winner`, `reason`, `stats`.
