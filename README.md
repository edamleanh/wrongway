# Wrongway Bot

Bot cho trò chơi **Wrongway** (đồ án AI Arena, HCMUS): bàn cờ 9×9, mỗi bên đi về hàng đối diện trước đối thủ, mỗi người có 10 bức tường để cản đường nhau.

Đề bài: <https://hcmus-cs-ta.github.io/lab-book/arena/statement/wrongway/>

## Bot hoạt động thế nào

File bot: [`wrongway-student/wrongway/submissions/MSSV.py`](wrongway-student/wrongway/submissions/MSSV.py). Đây là một file duy nhất, chỉ dùng thư viện chuẩn Python, đúng yêu cầu nộp bài.

- **Tìm kiếm alpha-beta (negamax)** với đào sâu dần và bảng chuyển vị. Trong 0.55 giây mỗi nước, bot nhìn trước được 5–6 nước.
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

### 1a. Đấu bot với bot, upload bot mới

Mở <http://localhost:8000/arena.html> (hoặc bấm **"Đấu bot vs bot →"** ở trang chính).

- **Chọn trận:** chọn bot **đi trước** (xuất phát ở dưới) và bot **đi sau** (xuất phát ở trên). Nút ⇅ đổi bên. Chọn thêm giới hạn thời gian mỗi nước (1, 2 hoặc 5 giây) và tốc độ xem, rồi bấm **Bắt đầu**.
- **Upload bot:** đặt tên (chữ không dấu, số, `_` hoặc `-`), chọn file `.py` có hàm `choose_action(observation)` giống bài nộp Arena (tối đa 100 KB), rồi bấm **Upload**. Bot mới xuất hiện trong danh sách cho mọi người.
- **Kết quả:** người thắng, số lượt, và thống kê từng bot (số nước hợp lệ, quá giờ, nước sai, lỗi, lần nghĩ lâu nhất). Nếu bot bị lỗi, traceback được hiện ra để sửa.

**Bot chạy ngay trong trình duyệt** bằng [Pyodide](https://pyodide.org) (Python trên WebAssembly), trong một Web Worker. Server chỉ lưu và trả mã nguồn, **không bao giờ chạy code người khác upload**. Luật lấy từ chính `environment.py` của Arena; bot lỗi, quá giờ hoặc trả nước sai thì mất lượt, giống Arena. Bot bị treo quá lâu sẽ bị ngắt và xử thua. Lần đầu mở trang cần tải khoảng 10 MB, và Python trên WebAssembly chậm hơn khoảng 2–3 lần, nên giới hạn mặc định là 2 giây mỗi nước.

Khi chạy cục bộ, bot upload được lưu trong thư mục `uploads/` (không đưa lên git).

### 1b. Deploy lên Vercel

Repo đã cấu hình sẵn để deploy:

1. Vào <https://vercel.com/new> và chọn **Import** repo này.
2. Framework Preset: **Other**. Để trống Build Command và Output Directory.
3. Bấm **Deploy**.

Hoặc dùng Vercel CLI: `npm i -g vercel`, rồi chạy `vercel` (bản thử) hoặc `vercel --prod` (bản chính).

**Bật kho lưu bot upload** (cần cho tính năng upload trên Vercel):

1. Trong project trên Vercel, vào tab **Storage**, bấm **Create Database**, chọn **Upstash for Redis** (gói Free).
2. Bấm **Connect** để nối database với project. Vercel tự thêm các biến môi trường `KV_REST_API_URL` và `KV_REST_API_TOKEN`.
3. (Không bắt buộc) Để xóa được bot, vào **Settings → Environment Variables** và thêm `ADMIN_PASSWORD` với mật khẩu bạn chọn.
4. Vào tab **Deployments** và bấm **Redeploy** để áp dụng biến môi trường.

Nếu chưa bật kho lưu, trang đấu bot vẫn chạy với 4 bot có sẵn, chỉ là chưa upload được.

Cách hoạt động: `public/index.html` là trang tĩnh. `api/move.py` là serverless function Python, dùng `api/_core.py` cùng Arena và bot. Server **không lưu trạng thái**: trình duyệt giữ ván cờ và gửi kèm mỗi request. Nhờ vậy nhiều người chơi cùng lúc được, mỗi người một ván riêng. Bản chạy cục bộ `tools/play_web.py` dùng chung đúng code này.

### 2. Cho hai bot đấu với nhau (Arena)

```bash
python wrongway-student/wrongway/arena.py -p1 MSSV -p2 wall_builder --render --log logs/demo.json
python wrongway-student/wrongway/replay.py logs/demo.json --delay 0.3
```

Các bot có sẵn trong `submissions/`: `MSSV`, `greedy`, `random_agent`, `wall_builder`. Đổi `-p1` và `-p2` để thử bot ở cả hai bên.

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
│       └── greedy.py, random_agent.py, wall_builder.py, TEMPLATE.py
├── public/
│   ├── index.html               # chơi với bot
│   ├── arena.html               # đấu bot vs bot, upload bot
│   └── match-worker.js          # chạy trận trong trình duyệt (Pyodide)
├── api/                         # Vercel: move.py (người vs bot), bots.py (thư viện bot)
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
