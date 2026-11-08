# Wrongway Arena agent

Đồ án AI Arena (HCMUS): viết agent Python cho trò Wrongway (bàn 9x9, đi tới hàng đối diện trước, mỗi bên 10 tường).
Đề: https://hcmus-cs-ta.github.io/lab-book/arena/statement/wrongway/ (bản đầy đủ: `wrongway-student/wrongway/PROBLEM_STATEMENT.md`, `STUDENT_GUIDE.md`).

## Cấu trúc

```
projject/
├── CLAUDE.md
├── .claude/skills/            # run-match, tournament, check-submission, wrongway-rules, train-model
├── wrongway-student/wrongway/ # Arena của giảng viên - KHÔNG sửa
│   ├── arena.py  environment.py  agent_runner.py  replay.py  visualizer.py
│   └── submissions/
│       ├── MSSV.py            # <-- BÀI CỦA MÌNH (đổi tên thành <mã số SV>.py khi nộp)
│       ├── TEMPLATE.py  greedy.py  random_agent.py  wall_builder.py
├── tools/
│   ├── tournament.py          # round-robin đổi bên, chấm điểm 3/1/0 như mùa giải
│   ├── check_submission.py    # kiểm tra trước khi nộp
│   └── play_web.py            # server cục bộ chơi người vs bot (dùng chung api/_core.py)
├── public/index.html          # giao diện bàn cờ SVG (trình duyệt giữ trạng thái ván)
├── api/                       # Vercel: move.py (serverless) + _core.py (logic stateless, dùng chung)
├── vercel.json, .vercelignore
├── training/                  # train ngoài file nộp (thư viện tùy ý)
│   ├── evolve.py              # tiến hóa (1+λ) qua tự đấu -> ghi WEIGHTS vào MSSV.py
│   ├── gauntlet.py            # đo sức mạnh 1 bot vs nhiều đối thủ (mở đầu ngẫu nhiên)
│   ├── opponents/             # bản cũ làm đối thủ chuẩn: v1, v2_untrained, v2_trained
│   ├── EXPERIMENTS.md         # nhật ký thử nghiệm: đã thử gì, kết quả, vì sao giữ/bỏ
│   └── common.py              # đường dẫn, ghi khối WEIGHTS
├── tests/test_bot.py          # luật trong bot khớp Arena; bot hợp lệ, đủ nhanh
└── logs/                      # replay JSON, log train (không commit)
```

## Thiết kế bot

Alpha-beta (negamax, đào sâu dần theo deadline) + hàm đánh giá tuyến tính. Trọng số được **học bằng tiến hóa qua tự đấu** (`training/evolve.py`) rồi **ghi thẳng vào khối `WEIGHTS BEGIN/END`** trong `MSSV.py` (file nộp không được đọc file ngoài). Luật mô phỏng trong bot phải khớp `environment.py` (có test). Chi tiết: skill `train-model`.

Thử nghiệm mới: copy MSSV.py thành `submissions/dev.py`, sửa, đo bằng `python training/gauntlet.py dev MSSV training/opponents/v2_trained.py --pairs 20 --workers 6` (40 ván/đối thủ, chênh < 0.1 là nhiễu), ghi kết quả vào `training/EXPERIMENTS.md`, xóa dev.py khi xong. Xem EXPERIMENTS.md trước để không thử lại cái đã bỏ.

## Ràng buộc bài nộp (bắt buộc)

- **Một file** `MSSV.py` độc lập, Python 3.9+, **chỉ thư viện chuẩn**. Không import từ Arena, không đọc file, không `input()`.
- `choose_action(observation)` trả về một trong: `{"type":"move","to":[r,c]}`, `{"type":"wall","orientation":"H"|"V","row":r,"col":c}`, `{"type":"pass"}`.
- Mỗi nước **≤ 1 s** (chấm trên Colab CPU, chậm hơn máy cá nhân → giữ ≤ ~0.5 s). Import ≤ 5 s. RAM ≤ 128 MB.
- Timeout/lỗi/hành động sai = mất lượt. Biến toàn cục có thể mất khi agent bị restart → phải chạy được chỉ từ `observation`.
- Tọa độ không lật khi đổi bên: luôn đọc `you["goal_row"]`.
- `legal_walls` không có khóa `"type"` → phải tự thêm `"type": "wall"`.

## Lệnh (chạy từ thư mục gốc `projject`; trên Windows dùng `py -3` hoặc `python`)

```
python wrongway-student/wrongway/arena.py -p1 MSSV -p2 wall_builder --render --log logs/demo.json
python wrongway-student/wrongway/replay.py logs/demo.json --delay 0.3
python tools/tournament.py MSSV --vs-all --seeds 3
python tools/check_submission.py wrongway-student/wrongway/submissions/MSSV.py
python -m unittest discover tests
python tools/play_web.py                 # chơi với bot trên trình duyệt (http://localhost:8000)
```

Không chạy nhiều trận song song khi đo thời gian (làm sai lệch timing).

## Quy ước khi sửa bot

- Mọi thay đổi chiến thuật: chạy `tests` + `tournament --vs-all` và so sánh trước/sau.
- Giữ khối `try/except` ở `choose_action` với nước dự phòng hợp lệ.
- Kiểm tra thời gian bằng `perf_counter()` với deadline (`TIME_BUDGET`) cho mọi vòng tìm kiếm.
- Sinh viên phải giải thích được code khi vấn đáp → giữ code dễ đọc, có chú thích.
