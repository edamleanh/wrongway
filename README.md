# Wrongway Bot

Bot cho trò chơi **Wrongway** (đồ án AI Arena, HCMUS): bàn cờ 9×9, mỗi bên đi về hàng đối diện trước đối thủ, mỗi người có 10 bức tường để cản đường nhau.

Đề bài: <https://hcmus-cs-ta.github.io/lab-book/arena/statement/wrongway/>

## Bot hoạt động thế nào

File bot: [`wrongway-student/wrongway/submissions/MSSV.py`](wrongway-student/wrongway/submissions/MSSV.py). Đây là một file duy nhất, chỉ dùng thư viện chuẩn Python, đúng yêu cầu nộp bài.

- **Tìm kiếm alpha-beta (negamax)** với đào sâu dần và bảng chuyển vị. Bot dùng 0.55 giây mỗi nước (giới hạn thật 1 giây, máy chấm chậm hơn), nhìn trước 5–6 nước trên laptop và khoảng 4–5 nước ở tốc độ máy chấm (tối đa 6).
- **Ứng viên mỗi nút**: mọi nước đi hợp lệ, cộng thêm 8 bức tường tốt nhất chắn đường ngắn nhất của đối thủ.
- **Hàm đánh giá tuyến tính** dựa trên khoảng cách tới đích của hai bên, số tường còn lại và lợi thế khi đối thủ đã hết tường. Các **trọng số được học bằng thuật toán tiến hóa qua tự đấu** ([`training/evolve.py`](training/evolve.py)), rồi ghi thẳng vào file bot.
- Bot tự mô phỏng luật chơi (đi, nhảy, đi chéo, tường hợp lệ). Test kiểm tra luật này khớp với `environment.py` của Arena trên hàng nghìn trạng thái ngẫu nhiên.

Kết quả đo được (40 ván mỗi đối thủ, đổi bên, có nước mở đầu ngẫu nhiên):

| Đối thủ | Điểm |
|---|---|
| `greedy` | 1.00 |
| `wall_builder` | 0.80–0.93 |
| Engine v1 (bản cũ) | 0.79–0.85 |

Các hướng đã thử (kể cả mạng nơ-ron bằng PyTorch) và lý do giữ hay bỏ: [`training/EXPERIMENTS.md`](training/EXPERIMENTS.md).

## Yêu cầu

- Python **3.9 trở lên**. Không cần cài thêm thư viện nào.
- Trên Windows dùng `python` hoặc `py -3`; trên macOS/Linux dùng `python3`.

Mọi lệnh bên dưới chạy từ **thư mục gốc của repo**.

## Hướng dẫn chạy

### 1. Chơi với bot trên trình duyệt

```bash
python tools/play_web.py
```

Trình duyệt tự mở <http://localhost:8000>. Nếu không, bạn mở link đó bằng tay.

- **Đi quân:** bấm vào ô có chấm xanh.
- **Đặt tường:** rê chuột vào khe giữa các ô (xanh là đặt được, đỏ là không) rồi bấm.
- **Chọn ván mới:** đi trước hoặc đi sau, và độ khó Dễ / Vừa / Khó. Mức Khó là đúng bot sẽ nộp.
- **↶ Đi lại:** huỷ nước vừa rồi của bạn cùng nước đáp của bot.

Tùy chọn khác: `--port 8080`, `--no-browser`, `--bot greedy` (đấu với một bot khác trong `submissions/`). Bấm `Ctrl+C` để dừng.

### 1a. Đấu bot với bot

Mở <http://localhost:8000/arena.html> (hoặc bấm **"Đấu bot vs bot →"** ở trang chính).

- **Chọn trận:** chọn bot **đi trước** (xuất phát ở dưới) và bot **đi sau** (xuất phát ở trên). Nút ⇅ đổi bên. Chọn thêm giới hạn thời gian mỗi nước (1, 2 hoặc 5 giây) và tốc độ xem, rồi bấm **Bắt đầu**.
- **Đấu nhiều trận:** chọn **Số trận** (1, 2, 6, 10, 20). Bật **Đổi bên sau mỗi trận** để hai bot lần lượt đi trước/đi sau (mỗi cặp trận dùng chung seed), giống mùa giải. Bảng điểm tính thắng 3, hòa 1, thua 0.
- **Lịch sử trận:** mỗi trận được lưu trong trình duyệt (tối đa 200 trận gần nhất). Bấm **Xem lại** để tua từng nước (nút ⏮ ◀ ▶ ⏭, thanh kéo, hoặc phím ← → và Space; Esc để thoát).
- **Kết quả:** người thắng, số lượt, và thống kê từng bot (số nước hợp lệ, quá giờ, nước sai, lỗi, lần nghĩ lâu nhất). Nếu bot bị lỗi, traceback được hiện ra để sửa.

**Bot chạy ngay trong trình duyệt** bằng [Pyodide](https://pyodide.org) (Python trên WebAssembly), trong một Web Worker. Server chỉ đọc và trả mã nguồn, **không bao giờ chạy code bot**. Luật lấy từ chính `environment.py` của Arena; bot lỗi, quá giờ hoặc trả nước sai thì mất lượt, giống Arena. Bot bị treo quá lâu sẽ bị ngắt và xử thua. Lần đầu mở trang cần tải khoảng 10 MB, và Python trên WebAssembly chậm hơn khoảng 2–3 lần, nên giới hạn mặc định là 2 giây mỗi nước.

### 1b. Thêm bot mới (qua git)

Web **không có upload**. Danh sách bot chính là các file trong [`wrongway-student/wrongway/submissions/`](wrongway-student/wrongway/submissions/):

1. Bỏ file `<TênBot>.py` vào `wrongway-student/wrongway/submissions/`.
   - Tên file 2–32 ký tự: chữ không dấu, số, `_` hoặc `-` (tên file là tên bot).
   - Có hàm `choose_action(observation)` ở cấp ngoài cùng, giống bài nộp Arena. Tối đa 100 KB.
2. (Nên có) Dòng đầu docstring của file là mô tả hiện trên web, ví dụ `"""Bot của Nam: alpha-beta độ sâu 3."""`.
3. Chạy thử trước khi push: `python wrongway-student/wrongway/arena.py -p1 <TênBot> -p2 greedy` (cột `error` phải bằng 0).
4. Commit và push lên nhánh `main`. Vercel tự deploy lại, bot mới hiện trong danh sách trang đấu bot sau khoảng 1 phút.

File lỗi cú pháp hoặc thiếu `choose_action` tự bị bỏ qua. Các tên `TEMPLATE`, `dev`, `bot` và file bắt đầu bằng `_` không được hiện (xem `HIDDEN` trong [`api/_bots.py`](api/_bots.py)). Muốn sửa hoặc xóa bot: sửa hoặc xóa file rồi push.

Cần quyền ghi vào repo để push; chủ repo mời thêm người ở **Settings → Collaborators** trên GitHub.

### 1c. Deploy lên Vercel

Repo đã cấu hình sẵn để deploy:

1. Vào <https://vercel.com/new> và chọn **Import** repo này.
2. Framework Preset: **Other**. Để trống Build Command và Output Directory.
3. Bấm **Deploy**.

Từ đó mỗi lần push lên `main`, Vercel tự deploy lại. Không cần database hay biến môi trường nào.

Cách hoạt động: `public/index.html` và `public/arena.html` là trang tĩnh. `api/move.py` (chơi người vs bot) là serverless function Python, dùng `api/_core.py` cùng Arena và bot. `api/bots.py` trả danh sách và mã nguồn bot trong `submissions/`. Server **không lưu trạng thái**: trình duyệt giữ ván cờ và gửi kèm mỗi request, nên nhiều người chơi cùng lúc được. Bản chạy cục bộ `tools/play_web.py` dùng chung đúng code này.

### 2. Cho hai bot đấu với nhau (Arena)

```bash
python wrongway-student/wrongway/arena.py -p1 MSSV -p2 wall_builder --render --log logs/demo.json
python wrongway-student/wrongway/replay.py logs/demo.json --delay 0.3
```

Mọi file trong `submissions/` đều dùng được làm bot (ví dụ `MSSV`, `HuynhDuy_test2`, `greedy`, `random_agent`, `wall_builder`). Đổi `-p1` và `-p2` để thử bot ở cả hai bên.

### 3. Giải đấu thử và đo sức mạnh

```bash
# Đấu vòng tròn kiểu mùa giải (mỗi cặp 2 ván đổi bên, thắng 3 hòa 1)
python tools/tournament.py MSSV --vs-all --seeds 3

# Đo một bot trước nhiều đối thủ, có nước mở đầu ngẫu nhiên (chạy song song)
python training/gauntlet.py MSSV training/opponents/v1.py wall_builder greedy --pairs 20 --workers 6
```

### 4. Kiểm tra

```bash
python -m unittest discover tests
python tools/check_submission.py wrongway-student/wrongway/submissions/MSSV.py
```

- **Unit test** kiểm tra luật trong bot khớp Arena, và bot luôn trả nước hợp lệ đủ nhanh.
- **`check_submission.py`** kiểm tra yêu cầu nộp bài: tên file, chỉ dùng thư viện chuẩn, có `choose_action`, cho bot đấu với các bot mẫu ở cả hai bên, và đo thời gian mỗi nước.

### 5. Train lại trọng số

```bash
python -u training/evolve.py --generations 30 --workers 10 > logs/evolve.log 2>&1
```

Mỗi thế hệ sinh vài bộ trọng số đột biến, cho chúng đấu với bản vô địch hiện tại và một nhóm đối thủ chuẩn (các nhà vô địch cũ, các bot trong `training/opponents/`, và `wall_builder`). Bộ nào thắng rõ ràng thì được ghi vào khối `WEIGHTS` trong `MSSV.py`. Bản trước mỗi lần thay được lưu ở `training/checkpoints/`. Đặt `--workers` bằng số luồng CPU bạn muốn dùng.

## Cấu trúc

```
├── wrongway-student/wrongway/   # Arena của giảng viên (luật, chạy trận, replay)
│   └── submissions/
│       ├── MSSV.py              # bot của mình
│       ├── HuynhDuy_test2.py, agent.py   # bot của người khác (thêm qua git)
│       └── greedy.py, random_agent.py, wall_builder.py, TEMPLATE.py
├── public/
│   ├── index.html               # chơi với bot
│   ├── arena.html               # đấu bot vs bot, lịch sử trận
│   └── match-worker.js          # chạy trận trong trình duyệt (Pyodide)
├── api/                         # Vercel: move.py (người vs bot), bots.py (danh sách bot từ submissions/)
├── vercel.json
├── tools/
│   ├── play_web.py              # chạy giao diện chơi cục bộ
│   ├── tournament.py            # đấu vòng tròn kiểu mùa giải
│   └── check_submission.py      # kiểm tra trước khi nộp
├── training/
│   ├── evolve.py                # học trọng số bằng tiến hóa
│   ├── gauntlet.py              # đo sức mạnh
│   ├── opponents/               # các bản bot cũ làm đối thủ chuẩn
│   └── EXPERIMENTS.md           # nhật ký thử nghiệm
├── tests/test_bot.py
├── CLAUDE.md, .claude/skills/   # hướng dẫn cho Claude Code
└── README.md
```

## Nộp bài

Đổi tên `MSSV.py` thành mã số sinh viên (ví dụ `26127000.py`) và nộp **đúng một file đó** lên Moodle.
