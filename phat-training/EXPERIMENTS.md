# Nhật ký thử nghiệm - phần của Phát

Nhật ký riêng cho việc buff `Phat_test2` -> `submissions/Phat_buffed_1.py` và hạ tầng đo của nó.
Tách khỏi `training/EXPERIMENTS.md` vì file đó còn ghi các đợt train của thành viên khác;
số thứ tự ở đây tiếp nối file đó (#47 trở đi), các chẩn đoán được nhắc tới (#44, #45) nằm trong file kia.

Điểm = (thắng + 0.5 × hòa) / số ván. Mỗi cặp ván cùng seed, đổi bên, 2 nước mở đầu ngẫu nhiên.
Lệnh đo ở tốc độ máy chấm (Colab chậm ~2 lần laptop):

    python phat-phat-training/gauntlet.py Phat_buffed_1 MSSV dtest2 az_agent --pairs 20 --workers 6 --budget 0.275 --seed 777

`--budget` / `--opp-budget` ghi đè `TIME_BUDGET` của từng bên (mặc định `--opp-budget` = `--budget`,
tức cả hai bên cùng chạy trên máy chậm hơn 2 lần). Không truyền cờ thì giữ nguyên số trong file bot.

## Buff `Phat_test2` -> `submissions/Phat_buffed_1.py`

Chẩn đoán #44/#45 nói Phat_test2 thua MSSV vì hàm đánh giá (không phải độ sâu). Thử hai hướng, đo bằng `phat-training/gauntlet.py` (16–28 ván/đối thủ, 2 nước mở đầu ngẫu nhiên, seed 12345 và 999; chênh dưới ~0.1 là nhiễu).

| # | Thử nghiệm | Kết quả | Quyết định |
|---|---|---|---|
| 47 | **Thay hàm đánh giá tra bảng** (bảng TD của `MSSV.py`) vào động cơ `Phat_test2`: lá rẻ (dist từ goal-layers, vuln 8 bước có cache + lười theo cửa sổ), ứng viên tường ở nút trong lấy từ đường ngắn nhất, thêm thưởng tiến bộ & "hết tường thì chỉ tiến" ở gốc | vs MSSV **0.19–0.25** (Phat_test2: 0.44), vs dtest2 0.44–0.47 (0.50), vs az_agent 0.25 (0.38), **vs Phat_test2 0.56–0.69** | **Bỏ**: cùng hệ đánh giá với MSSV thì thua chính nó nặng hơn (đấu gương), dù thắng dòng dtest2 |
| 48 | **Giữ hàm đánh giá tuyến tính, buff đúng chỗ yếu**: `WALL_SCALE` 1.35 cho `wall_diff`/`wall_quad` + lá rẻ (vuln 8 bước, cache, tính lười) + ứng viên tường từ đường ngắn nhất + killer/history chỉ xếp lại thứ tự (không chen vào top ứng viên) + thưởng tiến bộ & "hết tường thì chỉ tiến" ở gốc | vs dtest2 **0.72** (Phat_test2: 0.50), vs Phat_test2 **0.59**, vs MSSV **0.41 + 0.46** (0.44), vs az_agent 0.21–0.25 (0.38). `check_submission`: 6/6 thắng, 0 lỗi/luật sai, nước lâu nhất 0.560 s | **Dùng** -> `Phat_buffed_1.py` |
| 49 | `WALL_SCALE` 1.20 (giá trị #45 chọn) thay vì 1.35 | Cùng seed 999, 20 ván/đối thủ: MSSV **0.30**, az_agent 0.25, dtest2 0.62 | Giữ 1.35 |

Ghi chú: `az_agent` vẫn là đối thủ khó nhất (0.21–0.25) — nó là MSSV + policy ở gốc, hơn hẳn về chất lượng nước gốc. Hướng còn lại: tăng tốc tìm kiếm (bản này ~16.5k nút/s, MSSV ~22k nút/s) hoặc thêm policy/MCTS ở gốc như `az_agent`.

## Đo lại ở tốc độ máy chấm: hạ tầng `--budget` + WALL_SCALE 1.35 vs 1.20

Máy chấm (Colab) chậm hơn laptop ~2 lần, nên mọi số ở #47–#49 (đo ở 0.55 s/nước) có thể không còn đúng. `phat-training/gauntlet.py` nay có `--budget` / `--opp-budget` (docstring vẫn ghi từ trước nhưng argparse chưa có, nên lệnh `--budget` báo lỗi): ngân sách được đặt thẳng vào module của **từng bên** qua trường `side_budget` mới của `evolve.play`, vì tham số `budget` cũ chỉ áp cho bot dạng dict trọng số. Mặc định `--opp-budget` = `--budget`, tức mô phỏng "cả hai bên cùng chậm" như #45.

Quy ước đo: 0.275 s/nước (nửa ngân sách thật), cả hai bên, seed 12345 và 777, 20–30 cặp ván/đối thủ, mở đầu 2 nước ngẫu nhiên, `max_turns` 300. Mọi bản chạy cùng seed nên so sánh được theo từng cặp ván. Hai lưu ý về cách mô phỏng: hạ ngân sách chỉ làm bot nhìn nông hơn, **không** giống hệt CPU chậm 2 lần (chi phí cố định mỗi nước như đọc observation không đổi); và `max_turns` 300 (Arena cho 500) nên một phần ván với dtest2 bị cắt thành hòa.

| # | Thử nghiệm | Kết quả | Quyết định |
|---|---|---|---|
| 50 | **Hạ tầng**: thêm `--budget` / `--opp-budget` cho `phat-training/gauntlet.py` + trường `side_budget` cho `evolve.play` | 4 ván vs `greedy`: 11 s ở 0.55 s/nước → 1 s ở 0.05 s/nước, cùng kết quả từng ván. Không truyền cờ thì giữ nguyên `TIME_BUDGET` trong file (đường train của `evolve.py` không đổi) | **Dùng** |
| 51 | **WALL_SCALE 1.35 vs 1.20 ở tốc độ máy chấm** (0.275 s), 260 ván/bản (100 vs MSSV, 80 vs dtest2, 80 vs az_agent, 2 seed); kèm `Phat_test2` gốc làm mốc; đấu gương 1.35 vs 1.20 (40 ván) | Điểm (W–D–L): **1.35** — MSSV **0.22** (22-0-78), dtest2 **0.54** (37-12-31), az_agent **0.48** (38-1-41) → tổng **103.5/260 = 0.398**. **1.20** — MSSV **0.38** (38-0-62), dtest2 **0.36** (27-3-50), az_agent **0.46** (37-0-43) → tổng **103.5/260 = 0.398**. **`Phat_test2` gốc** — 0.215 / 0.238 / 0.181 → tổng **0.212**. Đấu gương: 1.35 được 0.47 (14-10-16) | **Hòa tuyệt đối về tổng, nhưng ngược nhau theo đối thủ** |

Đọc kết quả #51:

- **Cái buff là thật, cái hằng số thì không.** Ở tốc độ máy chấm, bản nào của `Phat_buffed_1` cũng gần **gấp đôi** `Phat_test2` gốc (0.40 vs 0.21). Nhưng 1.35 vs 1.20 chỉ là 103.5 vs 103.5 điểm trên 520 ván — đúng bằng nhau, không phải "1.35 thắng".
- **1.35 chỉ thắng đậm ở dòng dtest2** (0.54 vs 0.36; từng seed: 0.61/0.46 so với 0.44/0.28) và **thua đậm trước MSSV** (0.22 vs 0.38, ~2.5σ). Trước MSSV, 1.35 **không hơn gì bản gốc** (0.220 vs 0.215), còn 1.20 hơn rõ (+0.16).
- **Số 0.41–0.46 của #48 (vs MSSV ở 0.55 s) không tái hiện** ở 0.275 s: cả bản gốc lẫn 1.35 đều rơi về ~0.22. Lợi thế đó đến từ ngân sách thời gian, không từ WALL_SCALE.
- **Thứ tự đối thủ đổi theo tốc độ.** az_agent từ "khó nhất" (0.21–0.25 ở 0.55 s) thành dễ thắng hơn (0.46–0.48 ở 0.275 s) — suy luận: mạng policy của nó đắt theo mỗi nút nên mất độ sâu khi ngân sách giảm; MSSV ngược lại, mạnh lên tương đối (bản gốc 0.44 → 0.215). Kết luận "đối thủ nào khó" phải đo ở tốc độ máy chấm.
- **Nhiễu giữa seed vẫn lớn**: cùng một bot `Phat_test2` vs MSSV được 0.29 (seed 12345) và 0.17 (seed 777). Vì vậy mọi so sánh dưới đây gộp 2 seed và ≥ 40 ván/ô; một seed đơn lẻ vẫn đủ để lật kết luận.
- **Chọn hằng số**: tổng bằng nhau thì nên chọn theo worst case — 1.20 có sàn 0.36, 1.35 có sàn 0.22 (MSSV). Chưa đổi gì trong `Phat_buffed_1.py`; muốn ăn cả hai thì thử 1.25–1.28, hoặc để WALL_SCALE phụ thuộc đối thủ.

Log thô của đợt đo: `logs/wallscale_colab.log`, `logs/wallscale_base.log`, `logs/wallscale_seed777.log` (không commit).

## Hướng chưa ai thử: null-move pruning (ở đây `pass` là nước hợp lệ)

`training/EXPERIMENTS.md` loại các kỹ thuật "tìm không đều" sau #24 (LMR) và #30 (mở rộng tường nặng):
cả hai đều **giảm độ sâu của chính nước tường** nên mất chiến thuật. Null-move pruning khác hẳn: nó chỉ cắt
khi *nhường lượt* vẫn đủ tốt, không đụng tới độ sâu của nước tường thật. Và ở Wrongway `pass` là hành động
hợp lệ (PROBLEM_STATEMENT §2.3), nên "nhường lượt" là một phương án **thật** chứ không phải giả định như
trong cờ vua. Trong repo chưa ai thử (grep: không có `null`, `razor`, `aspiration` trong log).

| # | Thử nghiệm | Kết quả | Quyết định |
|---|---|---|---|
| 52 | **Null-move pruning** trong `negamax` của `Phat_buffed_1`: nếu `depth ≥ 3`, `ply > 0`, không phải nút ngay sau một null, `beta` không phải điểm thắng, bên đi còn tường → đổi bên không đi gì, tìm với `depth-1-R` (R = 2) cửa sổ hẹp; `≥ beta` thì cắt. Chỉ thêm 20 dòng, không đổi gì khác | **Độ sâu cố định 6, 20 thế cờ: 6.23 s → 2.30 s (nhanh 2.70×), chọn đúng cùng nước 20/20** → đây là tăng tốc thuần, không đổi quyết định. Ở 0.275 s/nước: độ sâu TB **5.45 → 6.95** (+1.5 tầng), số nút 2 295 → 3 395, cùng nước 17/20 (khác vì nhìn sâu hơn). **Nhưng sức mạnh không tăng**: đấu gương ở 0.275 s (cùng seed, cùng ngân sách) **0.45 / 48 ván** (15-13-20; seed 12345: 0.50, seed 777: 0.42); vs MSSV 16 ván **0.22** (3-1-12; bản gốc 0.22 / 100 ván ở #51) | **Không dùng ngay** — giữ làm đòn tăng tốc để thử tiếp |

Đọc kết quả #52:

- **Tăng tốc là thật và sạch.** Ở cùng độ sâu nó chọn y hệt 20/20 thế cờ, nên không phải "tìm ẩu": nó chỉ bỏ những nhánh mà ngay cả khi nhường lượt vẫn thắng. So với #34 (tăng tốc 2.6× bằng bitboard) thì đây là món tương đương mà chỉ tốn 20 dòng.
- **Nhưng thêm 1.5 tầng không thắng thêm ván nào.** Đấu gương 48 ván 0.45 — trong mức nhiễu, hơi âm. Cùng với #35/#37/#38 (thêm đặc trưng tĩnh không đổi gì) và #44/#45 (chẩn đoán là hàm đánh giá, không phải độ sâu), có nghĩa bot đang ở **cực tiểu cục bộ cả hai chiều**: sâu hơn và giàu đặc trưng hơn đều không tự động mạnh hơn.
- **Giả thuyết đáng thử nhất**: trọng số và `WALL_SCALE` được chọn/tiến hóa ở **độ sâu 4–6** (`#35`, `#47`–`#49`, và chính `evolve.py` chạy độ sâu cố định), nên khi bot tự dưng tìm tới 7–8 tầng thì chúng hết đúng — đúng như kết luận #36 ("bảng hợp với tìm kiếm nông, sai khi bot tìm sâu 6–7"). Tức là 2.7× tốc độ này **chỉ đáng giá nếu chọn lại hằng số ở độ sâu mới**.

Mã đã thử (chèn vào `negamax`, ngay trước khối `best_score = -INF / best_move = 0 / searched_tt = False`;
nhớ thêm `"null_block"` vào `__slots__` và `self.null_block = -1` trong `__init__`):

```python
if (NULL_R and depth >= 3 and ply > 0 and ply != self.null_block
        and beta < WIN_SCORE - 1000 and self.walls_left[me] > 0):
    head = self.null_block
    self.null_block = ply + 1
    self.side = opp
    self.zhash ^= ZOBRIST_SIDE
    v = -self.negamax(depth - 1 - NULL_R, -beta, -beta + 1, ply + 1)
    self.zhash ^= ZOBRIST_SIDE
    self.side = me
    self.null_block = head
    if self.stop:
        return 0.0
    if v >= beta:
        return v
```

## Hướng tiếp theo (theo thứ tự đáng làm)

1. **Chọn lại hằng số ở độ sâu mới, không phải thêm tầng.** Giữ null-move (miễn phí 2.7×), rồi quét `WALL_SCALE` 1.2 / 1.35 / 1.5 và `WALL_CANDIDATES_ROOT` 12 / 18 / 26 ở 0.275 s vs MSSV — những giá trị này chưa từng được đo khi bot tìm sâu 7–8. Nếu #51 đúng (1.35 vs 1.20 hòa vì cả hai đều bị chặn bởi hàm đánh giá), đây là chỗ duy nhất còn tiền.
2. **Tiêu 2.7× vào bề rộng chứ không phải độ sâu.** Cùng ngân sách, tăng `WALL_CANDIDATES_ROOT` 18 → 30 (`#9` chỉ thử ở nút trong, gốc chưa thử) — đúng chỗ #25 nói bot bỏ sót bẫy nhiều tường.
3. **Tăng tốc thuần cho chính bot này** (chưa ai profile): 16.5k nút/s so với 22k của MSSV; #34 cho MSSV thấy 2.6× tốc độ = từ 0.52 lên 0.75 vs AI Hard, và nó không đổi nước đi nên không có rủi ro "khớp độ sâu".
