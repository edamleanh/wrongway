---
name: tournament
description: Chạy round-robin Wrongway giống mùa giải (mỗi cặp 2 ván đổi bên, thắng 3 hòa 1) để đánh giá sức mạnh bot, so sánh phiên bản trước/sau khi sửa chiến thuật. Dùng khi người dùng hỏi bot mạnh cỡ nào, muốn benchmark, so sánh, hoặc sau mỗi thay đổi chiến thuật.
---

# Giải đấu thử

```
python tools/tournament.py MSSV --vs-all --seeds 3          # bot của mình vs mọi bot trong submissions/
python tools/tournament.py MSSV greedy wall_builder --seeds 2  # round-robin đầy đủ giữa các bot liệt kê
```

Tùy chọn: `--seeds N` (mỗi seed = 2 ván đổi bên), `--save-logs` (lưu replay vào `logs/`), `--timeout`, `--max-turns`.

## So sánh hai phiên bản bot

1. Sao lưu phiên bản cũ: copy `submissions/MSSV.py` thành `submissions/MSSV_old.py` (không nộp file này).
2. Sửa `MSSV.py`.
3. `python tools/tournament.py MSSV MSSV_old greedy random_agent wall_builder --seeds 3`.
4. Báo cáo bảng điểm. Cột `Ratio` = `(3W + D) / (3 x số ván)`, đúng công thức điểm mùa giải. Chú ý cột `TO/Inv/Err` phải bằng 0 và `MaxT(s)` nên < 0.5 s.
5. Xóa `MSSV_old.py` khi xong nếu không cần nữa (hỏi người dùng trước).

Lưu ý: chạy tuần tự, không chạy nhiều lệnh tournament song song (làm sai thời gian và có thể gây timeout giả).
