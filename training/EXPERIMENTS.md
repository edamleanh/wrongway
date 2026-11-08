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

**Kết luận đợt v3:** giữ nguyên v2 trong `MSSV.py`. Bài học: tối ưu theo một đối thủ duy nhất dễ khớp riêng với đối thủ đó; nếu tiến hóa tiếp nên dùng mục tiêu là trung bình trên nhiều đối thủ (AI Hard + các bản cũ).

Ghi chú về luật: quá 1 giây cũng làm quân đứng yên, nhưng Arena giết tiến trình bot (mất kết quả đang nghĩ và biến toàn cục), nên luôn trả `{"type": "pass"}` thay vì để quá giờ.

## Điểm yếu đã biết
- Hiệu ứng chân trời: đôi khi đi qua đi lại để trì hoãn một bức tường nguy hiểm của đối thủ (thử nghiệm 7, 8 chưa sửa được mà không mất sức mạnh).
- Chỉ xét tường chắn đường ngắn nhất của đối thủ; không có tường "phòng thủ".

## Ý tưởng chưa thử
- Đặc trưng "độ bền đường đi": số bước tăng thêm nếu đối thủ đặt bức tường tốt nhất (đắt, cần đo độ sâu bị mất).
- Tiến hóa ở độ sâu 5 cho sát lúc nộp; tiến hóa đồng thời cả `WALL_CANDIDATES`.
