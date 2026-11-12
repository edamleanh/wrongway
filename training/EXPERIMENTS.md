# Nhật ký thử nghiệm

Điểm = (thắng + 0.5 × hòa) / số ván. Mỗi cặp ván cùng seed, đổi bên, 2 nước mở đầu ngẫu nhiên (`training/gauntlet.py`). 40 ván → sai số khoảng ±0.08, nên chênh lệch dưới ~0.1 coi là nhiễu.

| # | Thử nghiệm | Kết quả | Quyết định |
|---|---|---|---|
| 1 | Greedy theo `distances` + đặt tường heuristic | Thua `wall_builder` | Điểm khởi đầu |
| 2 | Value network (PyTorch, MLP 315→64→32→1) học từ tự đấu, chọn nước 1-ply | Thua bot heuristic 0–5% sau 4 vòng. Model học tương quan giả "đặt tường = thắng" (dữ liệu từ bot chỉ đặt tường khi có lợi) | **Bỏ** |
| 3 | Engine v1: alpha-beta, đào sâu dần, ô cờ dạng tuple, độ sâu ≤ 4 | Thắng greedy/random, thua `wall_builder` (xài hết tường sớm) | Giữ làm đối thủ chuẩn `opponents/v1.py` |
| 4 | Tiến hóa trọng số trên v1 (10 thế hệ) | `walls` 0.3 → 1.8: học được việc giữ tường | Lấy trọng số sang v2 |
| 5 | Engine v2: ô cờ số nguyên, cạnh chặn dạng frozenset, bảng chuyển vị → độ sâu 5–6 | Cùng trọng số: **0.75 vs v1**, 0.88 vs `wall_builder` | **Dùng** |
| 6 | Tiến hóa trên v2 (30 thế hệ, độ sâu cố định 4, hall of fame) | Bản cuối: **0.85 vs v1, 0.60 vs v2 chưa train**, 0.80–0.93 vs `wall_builder`, 1.00 vs greedy | **Dùng** → `opponents/v2_trained.py` |
| 7 | Quiescence tường (ở lá xét bức tường tốt nhất) chống hiệu ứng chân trời | 0.62 rồi 0.51 vs v2_trained; kém hơn trước v1 (tốn độ sâu) | Bỏ |
| 8 | Phạt quay lại ô vừa đứng (chống đi qua đi lại) | 0.45 vs v2_trained; tốt hơn với `wall_builder` nhưng kém với bot mạnh | Bỏ |
| 9 | `WALL_CANDIDATES` 12 / 5 / 3 (mặc định 8) | 0.42 / ~0.51 / 0.38 | Giữ 8 |
| 10 | `TIME_BUDGET` 0.8 s (mặc định 0.55 s) | 0.51: thêm 0.25 s hiếm khi đủ cho một tầng sâu hơn | Giữ 0.55 s (an toàn hơn) |

| 11 | Đối thủ mới: AI Hard của wrongway.app, chạy offline với sự cho phép của tác giả (chỉ kiểm thử riêng, không đưa code vào repo) | v2_trained thắng 7/30 và 5/30 (**0.17–0.23**); thua nhanh khi đi sau vì đặt tường quá muộn | Dùng làm mục tiêu |
| 12 | Thêm nước `pass` (đứng yên, hợp lệ theo luật) vào **mọi nút** tìm kiếm | 0.30 vs AI Hard nhưng **0.30 vs v2_trained** (cây phình to, mất độ sâu) | Bỏ |
| 13 | `pass` **chỉ ở gốc** (cho mình, không làm phình cây) → engine v3 | Lần đầu 0.27 vs AI Hard, 0.58 vs v2_trained. **Đo lại (seed 777, 40 ván): 0.26 vs AI Hard, 0.50 vs v2**, v2 cũng được 0.28 vs AI Hard trên cùng seed → lợi thế lần đầu chỉ là nhiễu | Bỏ, giữ làm đối thủ `opponents/v3_pass_root.py` |
| 14 | Tiến hóa v3 (20 thế hệ, độ sâu 4) chọn đột biến **theo điểm vs AI Hard**, thêm trọng số `pass_penalty` (học được ≈ 0.47) | 0.33 vs AI Hard (v2: 0.28, trong mức nhiễu) nhưng **0.28 vs v2**, 0.34 vs v3 gốc (17 hòa) → khớp riêng với AI Hard, yếu trước bot alpha-beta | Bỏ, giữ làm đối thủ `opponents/v3_evolved.py` |

| 15 | Đặc trưng `vuln` (độ dễ bị chặn: bức tường tốt nhất trên 4 bước đầu của đường đi làm người đó chậm thêm bao nhiêu), tính ở lá, có cache. Phân tích ván thua AI Hard cho thấy bot hay đi vào chỗ một bức tường làm đường dài thêm ~8 bước | Trọng số 0.4 / 0.8 / 1.2 / 1.6: **0.44 / 0.49 / 0.49 / 0.32 vs AI Hard** (80 ván, v2 cùng seed 0.28). Xét cả đường đi (thay vì 4 bước): 0.42. Nhưng `vuln` 1.0 chỉ 0.34 vs v1 (đi qua đi lại, hòa nhiều) | Dùng cùng #16 |
| 16 | **Tường phòng thủ** ở gốc: tìm tối đa 3 bức tường đối thủ có thể đặt làm mình chậm ≥ 2 bước, xét thêm tường của mình chồng lấn/cắt ngang chỗ đó (không làm mình chậm đi). Lý do: bot `HuynhDuy_test2` (dựng hàng tường + cột tường để nhốt) thắng v2 **cả hai ván ở thế xuất phát chuẩn** dù v2 thắng nó 0.80 khi mở đầu ngẫu nhiên | Cùng `vuln` 1.0: **0.52 vs AI Hard (120 ván)**, **0.74 vs v2** (80 ván), 0.90 vs HuynhDuy (thắng cả 2 ván thế chuẩn), ~0.51 vs v1. Biến thể `vuln` 0.6: 0.465 vs Hard; thêm phạt đi qua đi lại: 0.435 vs Hard | **Dùng** → `MSSV.py` (v4) |
| 17 | Tiến hóa lại v4 (12 thế hệ, độ sâu 4) với mục tiêu **trung bình 4 đối thủ** (AI Hard, HuynhDuy, v1, v2), nhận 4 bộ trọng số (`safe_race` 6.85→8.0, `wall_urgency` -0.08→-0.51, `opp_dist` -0.42→+0.12) | Đo bằng thời gian thật, seed 2026: **0.31 vs AI Hard** (80 ván; v4 cùng seed 0.50) nhưng 0.65 vs v4, 0.68 vs v1, 0.68 vs v2, 0.95 vs HuynhDuy → mạnh hơn trước bot alpha-beta, yếu hẳn trước AI Hard | Không thay `MSSV.py` (mục tiêu là AI Hard); giữ làm đối thủ `opponents/v5_evolved.py` |
| 18 | Nhìn sâu hơn, đo ở **tốc độ Colab** (mô phỏng: ngân sách thời gian chia 2, vì máy chấm chậm hơn i5-11400H khoảng 2 lần). Đo độ sâu: với trần 6, khoảng 40% số nước còn thời gian để sâu hơn (cuối ván); vượt ngân sách tối đa 15–38 ms. Biến thể: bỏ trần (MAX_DEPTH 12) với 0.55 / 0.70 / 0.80 s | Seed 2026 vs AI Hard: bản hiện tại 0.41, trần 12: 0.53 / 0.44 / 0.60; đấu trực tiếp với bản hiện tại 0.59 / 0.42 / 0.51. **Seed 777: trần 12 chỉ 0.39 vs 0.36, đấu trực tiếp 0.45**. Thế chuẩn: trần 12 **thua `wall_builder` cả 2 ván** (bản hiện tại thắng 2/2), hòa 1-1 với `agent`. 0.80 s: lần nghĩ lâu nhất ≈ 0.88 s trên Colab chưa tính giao tiếp → quá sát giới hạn 1 s | **Bỏ**: không mạnh hơn rõ, lại thua ván thế chuẩn. Giữ 0.55 s, trần 6 |
| 19 | **Bitboard** cho BFS khoảng cách (profile: `distance()` chiếm 70% thời gian). Bàn = số nguyên 81 bit, 4 bitboard "đi được" (lên/xuống/trái/phải) đi kèm tập cạnh chặn (`class Blocked(frozenset)`), mỗi bước BFS = 4 phép dịch bit | **Nhanh 2.05×, chọn đúng từng nước như cũ** ở độ sâu 4–5 (cùng hash seed; 3000 bàn ngẫu nhiên khớp khoảng cách). Tốc độ Colab: 0.45 / 0.44 vs AI Hard (bản cũ 0.41 / 0.36), 0.48 đấu trực tiếp | Dùng |
| 20 | Dùng **kết quả độ sâu dở dang** khi hết giờ: nước đã vượt nước tốt nhất cũ (được xét trước) thì dùng luôn | Tốc độ Colab cùng bitboard: 0.49 / 0.36 vs AI Hard, **0.54 vs bản cũ**, 0.51 vs chỉ-bitboard; thế chuẩn thắng hết HuynhDuy, agent, wall_builder, greedy | **Dùng** (cùng #19) → `MSSV.py` |
| 21 | **Sắp thứ tự nước kiểu Stockfish** (cờ bật/tắt riêng): killer move (2/tầng), history heuristic (khi hòa điểm tĩnh), PVS (cửa sổ hẹp), **TT move trước khi sinh nước con** (`make_child`) | Độ sâu cố định, 45 thế cờ: độ sâu 5 nhanh **1.26×**, độ sâu 6 **1.32×** (số nút −12%), **45/45 cùng nước**. Riêng killer+history+PVS chỉ 1.07–1.09× (thứ tự tĩnh theo lợi ích tường vốn đã tốt); phần lớn nhờ TT-trước-khi-sinh (bỏ BFS cho mọi tường khi cắt tỉa ngay). Tốc độ Colab, 160 ván mở đầu ngẫu nhiên: **0.60 vs MSSV #20** (94-3-63), 0.43 vs AI Hard (không đổi). Thế chuẩn 20 ván: vs agent **1.00** (bản #20: 0.72), vs AI Hard 0.35 (như cũ), vs bản #20 0.47, HuynhDuy 1.00 | **Dùng** → `MSSV.py` |

Ghi chú: thứ tự xét các tường bằng điểm phụ thuộc hash chuỗi ngẫu nhiên của Python (`set` trong `walls_on_path`), nên cùng một bot có thể đánh hơi khác giữa các lần chạy; so sánh 'cùng nước' phải cố định `PYTHONHASHSEED`.

**Bài học:** bot tất định có thể bị một chuỗi nước cố định "bắt bài" ở thế xuất phát chuẩn mà đo bằng mở đầu ngẫu nhiên không thấy → luôn kiểm tra thêm cả hai ván thế chuẩn với từng đối thủ mới.

**Kết luận đợt v3:** giữ nguyên v2 trong `MSSV.py`. Bài học: tối ưu theo một đối thủ duy nhất dễ khớp riêng với đối thủ đó; nếu tiến hóa tiếp nên dùng mục tiêu là trung bình trên nhiều đối thủ (AI Hard + các bản cũ).

Ghi chú về luật: quá 1 giây cũng làm quân đứng yên, nhưng Arena giết tiến trình bot (mất kết quả đang nghĩ và biến toàn cục), nên luôn trả `{"type": "pass"}` thay vì để quá giờ.

## Đối chiếu với bot bên ngoài (MSSV v4, 0.55 s/nước)

| Đối thủ | Cách đấu | Kết quả |
|---|---|---|
| AI MCTS của [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) (MIT, Quoridor = luật Wrongway trừ `pass`), chạy offline qua Node, chỉ kiểm thử riêng | Novice 2.5k rollout (~1 s/nước): 40 ván mở đầu ngẫu nhiên / 40 ván thế chuẩn | **0.97 / 0.95** |
| | Average 7.5k rollout (~3 s/nước): 20 ván mở đầu ngẫu nhiên | **0.65** |
| | Strong 60k rollout (~21 s/nước): 2 ván thế chuẩn | 2/2 thắng |
| `agent` (bot bạn gửi, alpha-beta, sửa bộ đọc tường cho khớp Arena) | 2 ván thế chuẩn / 40 ván mở đầu ngẫu nhiên | 2/2 thắng / **0.81** (3 hòa) |
| AI Hard (wrongway.app, offline, có phép) | 80 ván mở đầu ngẫu nhiên, seed 2026 | 0.50 |

## Điểm yếu đã biết
- Hiệu ứng chân trời: đôi khi đi qua đi lại để trì hoãn một bức tường nguy hiểm của đối thủ (thử nghiệm 7, 8 chưa sửa được mà không mất sức mạnh).
- Tường phòng thủ chỉ xét ở gốc, chỉ chống bẫy một bức tường (bẫy cần nhiều tường có thể vẫn lọt).
- `vuln` làm bot hơi rụt rè trước bot alpha-beta yếu (v1 chỉ ~0.5).
- Trọng số tốt trước bot alpha-beta và tốt trước AI Hard đang kéo ngược nhau (thử nghiệm 17); điểm tiến hóa ở độ sâu 4 không phản ánh đúng sức mạnh khi chạy theo thời gian thật trước AI Hard.

## Ý tưởng chưa thử
- Tiến hóa ở độ sâu 5 cho sát lúc nộp; tiến hóa đồng thời cả `WALL_CANDIDATES`.
- Đo lại mọi thay đổi ở tốc độ Colab (ngân sách / 2): ở tốc độ đó v4 chỉ ~0.39 vs AI Hard (thay vì 0.50) và thua cả 2 ván thế chuẩn với AI Hard.
