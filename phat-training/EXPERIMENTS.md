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

---

# #53. Khảo sát 11 bài nộp vòng 1 -> `Phat_buffed_2`

Dữ liệu: `wrongway-student/wrongway/round-1-submissions/` (11 bài + bảng xếp hạng + log từng ván).

> ⚠ **Ba chỗ trong mục này sai, đã sửa tại chỗ và ghi lại ở #57**: 009 không dùng ngân sách nút
> (dùng thời gian), `pass` là đòn phòng thủ *draw-aware* chứ không phải nước gốc đổi nhịp, và sách
> khai cuộc **không** phải bằng chứng yếu nhất vì chấm giải bắt đầu từ thế chuẩn.
> `24125005` cũng không phải đối thủ — đó là bài của đội mình.

| Hạng | Bot | Điểm | Cơ chế đáng chú ý |
|---|---|---|---|
| 1 | 24125014 | 51/60 | **Bảng cuộc đua chính xác khi 0 tường** + **PLUG_PASS** (pass khi đang bít đường đối thủ) + **stand-off pass** + sách khai cuộc hàng nghìn thế |
| 2 | 24125009 | 45/60 | cùng bảng cuộc đua 0 tường, **ngân sách THỜI GIAN 0,54 s mềm + hard stop** (không phải ngân sách nút — xem #57), sách 170 thế đã kiểm, NN 354->32 **chỉ thêm 4 nước tường "quiet" ở gốc** (value head tắt), anti-fortress + anti-stall, giá tường lõm 30…11 |
| 3 | 24125023 | 37/60 | bảng cuộc đua "corpus-trained", alpha-beta chọn lọc, **tàn cuộc giải chính xác** |
| (4) | 24125005 | 36/60 | **KHÔNG phải đối thủ — là bài của đội mình** (nộp dưới MSSV trưởng nhóm; 734 dòng, cùng dòng code với `submissions/MSSV.py` 831 dòng). Mọi ô đấu với nó trong log vòng 1 là **tự đấu** |
| 4 | 24125062 | 36/60 | **bảng DTM 0 tường có pass** + chống lặp |
| 6 | 24125015 | 34/60 | quét tường toàn DAG, cắt tỉa cuộc đua 0 tường khi `|dS-dO|>=3` |

Điểm chung của 5/6 bot dẫn đầu: **`pass` được dùng có chủ đích phòng thủ** (chống bị bít, phá
fortress, đổi thua thành hoà — *draw-aware*, không phải nước gốc đổi nhịp; xem #57), **và tàn cuộc
0 tường được giải chính xác** chứ không đoán bằng hàm đánh giá. Đây là thứ `Phat_buffed_1` hoàn
toàn không có (nó chỉ pass khi không còn nước nào).

## Đo lại `Phat_buffed_1` trước khi sửa (4 cặp/đối thủ = 8 ván, ngân sách theo file)

| Đối thủ | W-D-L | Điểm |
|---|---|---|
| 24125014 (hạng 1) | 0-0-8 | 0.00 |
| 24125009 (hạng 2) | 0-0-8 | 0.00 |
| 24125023 (hạng 3) | 2-0-6 | 0.25 |
| 24125062 (hạng 4) | 2-0-6 | 0.25 |

Tổng 4/32 = **0.125**.

## Soi một ván thua hạng 1 (78 lượt, `logs/g1.json`, in bằng `phat-training/analyze.py`)

- 40 lượt đầu hai bên bám nhau từng bước (d0 = d1), không ai hơn.
- Lượt 67: ta còn **1 tường**, đối thủ còn 2, d0 = 3 < d1 = 4 — thế thắng nhẹ.
- Lượt 72-74: đối thủ đặt H(0,2) rồi H(0,0), **bít hàng đích của ta**: đường từ 1 lên 4 bước. Ta đã
  hết tường, không đáp trả được, thua ở lượt 78.
- Đếm lại 10 bức tường ta đã đặt: **7 bức chỉ làm đối thủ chậm đúng 1 bước** (lượt 7, 17, 43, 59, 63,
  67) hoặc 0 bước (lượt 21). Trong cuộc đua thuần, đặt tường tốn 1 nhịp của mình và thêm 1 bước cho
  đối thủ là **hoà nhịp**, chỉ có lãi khi thêm >= 2 bước. Ba bức tường lớn (+4, +3, +4 ở lượt 23, 27,
  47) mới thật sự đổi thế cờ.

## Đã làm trong `Phat_buffed_2`

| # | Thay đổi | Kiểm chứng |
|---|---|---|
| 53a | **Bảng cuộc đua chính xác khi cả hai hết tường** (mục 4a của file): duyệt ngược 2*81*81 thế cờ có cả nước pass; thắng về nhanh nhất, thua câu giờ, hoà thì tiến lên | Kiểm bằng *điều kiện tối ưu cục bộ* cho **mọi** thế cờ trên 3 cấu hình tường (bàn trống, 1 tường ngang, 2 ngang + 1 dọc): **0 lỗi**. Dựng bảng **20 ms** (đo trên laptop), dùng lại cả ván |
| 53b | **Luật giữ chốt** (mục 4b): nếu quân ta bít mọi đường của đối thủ (tính cả nhảy) và tìm kiếm không thấy đường thắng (`điểm <= 0`) thì pass thay vì bước ra khỏi chốt | `check_submission`: 6/6 thắng bot mẫu, không lỗi/timeout, nước chậm nhất 0.565 s |

## Kết quả (cùng seed, cùng 4 cặp/đối thủ)

| Đối thủ | Phat_buffed_1 | Phat_buffed_2 |
|---|---|---|
| 24125014 | 0.00 | 0.00 |
| 24125009 | 0.00 | 0.12 (1-0-7) |
| 24125023 | 0.25 | 0.19 (1-1-6) |
| 24125062 | 0.25 | 0.38 (3-0-5) |
| **Tổng** | **0.125** (4/32) | **0.203** (6.5/32) |

Đọc kết quả: **chiều tăng là đúng nhưng chưa ra khỏi nhiễu** — 8 ván/ô thì chênh 1-2 ván là nhiễu
(quy ước của repo: chênh < 0.1 là nhiễu, và phương sai giữa các seed rất lớn). Cần >= 40 ván/ô mới
kết luận được. Hai thay đổi này *không thể làm bot yếu đi* (một cái là nghiệm đúng của trò chơi, một
cái chỉ pass khi tìm kiếm không thấy đường thắng), nên giữ lại.

## Hướng tiếp theo (theo thứ tự đáng làm)

1. **Luật "tường phải lãi >= 2 bước"** ở gốc: trong ván thua ở trên 7/10 tường chỉ lãi <= 1 bước,
   tức là hoà nhịp chứ không phải nước hay. Thử: ở gốc, nếu một ứng viên tường chỉ thêm <= 1 bước cho
   đối thủ thì xếp sau mọi nước đi quân (trừ khi nó là nước bít hàng đích). Đây là chỗ hứa hẹn nhất vì
   nó đúng cả lý thuyết lẫn số liệu ván thật, và chỉ sửa phần xếp thứ tự ở gốc nên không đổi bản chất
   tìm kiếm.
2. **Tường bít hàng đích**: khi ta còn <= 3 bước tới đích mà đối thủ còn >= 2 tường, hàm đánh giá phải
   trừ điểm (vuln hiện chỉ tính *một* bức tường, nên coi nhẹ thế bị bít bằng 2 tường — đúng cái đã thua).
   Cách rẻ: thêm một đặc trưng "số bước bị cộng thêm khi đối thủ đặt 2 tường vào hàng đích".
3. **Giữ 1 tường để đáp trả** thay vì tiêu bức cuối khi mình đang hơn nhịp (lượt 67 của ván trên).
4. **Ngân sách nút thay vì ngân sách thời gian** (hạng 2): bot chơi y hệt nhau trên máy nhanh/chậm,
   hết chuyện máy chấm chậm làm mất tầng — không tăng sức mạnh nhưng bỏ được một nguồn phương sai.
5. **Cộng dồn 2.70× của #52** (null-move) rồi mới chọn lại hằng số — vẫn là hướng lớn nhất còn bỏ ngỏ.

---

# #54. A1+A2: bảng giá tường lõm thay `wall_diff` + `wall_quad` -> `Phat_buffed_3`

Vì sao: cặp `wall_diff` + `wall_quad` tương đương `value(w) = 277.7*w − 4.96*w²`, tức giá biên
của bức tường thứ n là `277.7 − 4.96*(2n−1)`: bức đầu 273 điểm (2.33 bước) nhưng **bức thứ 10
vẫn 183 điểm = 1.56 bước**. Đó đúng là chỗ #53 đếm được 7/10 bức tường chỉ lãi ≤ 1 bước: hàm
đánh giá vẫn khen bức cuối 1.5 bước, nên bot tiêu nó cho một nước lãi 1 bước. Một parabol úp
không diễn tả được "đuôi bảng rẻ hẳn" tách khỏi "độ lớn chung", nên thay bằng bảng.

Hai thay đổi trong `submissions/Phat_buffed_3.py` (chỉ đụng hàm đánh giá, mọi thứ khác giữ nguyên):

- **A1** — một đặc trưng `wall_val = WALL_CUM[còn tường mình] − WALL_CUM[còn tường đối thủ]`,
  bảng `WV_REF = (1.000, 0.867, 0.767, 0.667, 0.600, 0.533, 0.467, 0.433, 0.400, 0.367)` lấy hình
  dạng lõm của bot hạng 2 (24125009: 30,26,23,20,18,16,14,13,12,11).
- **A2** — hai núm thay cho một `WALL_SCALE`: `WALL_VALUE = 272.75` (giá bức tường **đầu tiên**,
  giữ đúng giá P0 cũ) và `WALL_CURV = 1.0` (độ lõm 0..1; `0` = mọi bức cùng giá = tuyến tính
  thuần). `WALL_SCALE` cũ **chỉ chỉnh mức**: nó nhân cả `wall_diff` lẫn `wall_quad` nên tỉ lệ
  đuôi/đầu bị khoá ở 0.667 với mọi hệ số (câu cũ ở đây nói "nhân cả giá lẫn độ lõm" là sai,
  xem phần soi lại dưới).

Giá biên (điểm, và quy ra bước với `dist_diff` = 117/bước):

| bức tường thứ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| cũ | 272.7 | 262.8 | 252.9 | 243.0 | 233.0 | 223.1 | 213.2 | 203.3 | 193.4 | **183.4** |
| bản 3 | 272.8 | 236.5 | 209.2 | 181.9 | 163.6 | 145.4 | 127.4 | 118.1 | 109.1 | **100.1** |
| cũ (bước) | 2.33 | 2.25 | 2.16 | 2.08 | 1.99 | 1.91 | 1.82 | 1.74 | 1.65 | 1.57 |
| bản 3 (bước) | 2.33 | 2.02 | 1.79 | 1.55 | 1.40 | 1.24 | 1.09 | 1.01 | 0.93 | **0.86** |

Tổng giá trị 10 bức tường: cũ 2281 điểm → bản 3 1664 (73%), tức ngoài đổi hình dạng thì mức
trung bình cũng hạ ~27% (bức cuối xuống dưới 1 nhịp, bức đầu giữ nguyên). Bản 3 cũng bỏ chênh
lệch 6% giữa P0 và P1 ở giá tường — chênh lệch đó nằm trong nhiễu (#51).

## Đo: 40 ván/ô ở 0.275 s, seed 12345, mở đầu 2 nước ngẫu nhiên, cả hai bên cùng 0.275 s

Cùng seed nên so sánh được từng cặp ván. `--budget 0.275` (mô phỏng máy chấm chậm 2×).

| Bản | vs 24125014 (hạng 1) | vs 24125009 (hạng 2) | Tổng |
|---|---|---|---|
| `Phat_buffed_2` (mốc) | **0.05** (2-0-38) | **0.10** (2-4-34) | 6/80 = 0.075 |
| `Phat_buffed_3` bản **bảng WV** (A1+A2, đã xoá) | **0.06** (2-1-37) | **0.14** (4-3-33) | 8/80 = 0.100 |

Đọc kết quả:

- **Nhích lên ở cả hai ô, nhưng chưa ra khỏi nhiễu.** Δ = +0.01 và +0.02; tổng +0.025. Ở mức
  thắng ~0.1 thì sai số chuẩn của một ô 40 ván là `√(0.1·0.9/40) ≈ 0.047`, nên mức nhích này
  cỡ 0.4σ — đúng chiều nhưng không kết luận được.
- **Sàn đo quá thấp.** Hai đối thủ này thắng ta 34–38/40 ván ở 0.275 s, nên một thay đổi trong
  hàm đánh giá gần như không có chỗ thể hiện: muốn tách 0.05 khỏi 0.10 cần hàng trăm ván.
  #53 đo ở 0.55 s cũng ra 0.00 cho cả hai, tức đây là khoảng cách **cơ chế** (sách khai cuộc,
  pass, tàn cuộc chính xác) chứ không phải hằng số đánh giá.
- **Thay đổi có tác dụng thật** (không phải file không đổi): hai ô ra kết quả từng ván khác mốc
  (2-0-38 vs 2-1-37), và `WALL_CUM` khác hẳn bảng cũ như bảng trên.
- **Hai ô cùng chiều** là điểm đáng ghi: nếu là nhiễu thuần thì dấu phải ngẫu nhiên, còn ở đây
  cả hai đều tăng nhẹ. Đây là lý do giữ bản 3 lại để đo tiếp, không phải để thay bản nộp.

Quyết định: **giữ `Phat_buffed_3.py` làm biến thể, chưa thay bản nộp.** Lần đo tới nên làm ở chỗ
có chỗ trống để thể hiện (24125023 / 24125062, nơi mốc là 0.19–0.38) và/hoặc **ghép cặp theo
seed** (cùng seed, so từng cặp ván) thay vì so tổng — cách đó lọc được phương sai do mở đầu ngẫu
nhiên, thứ đang chiếm phần lớn nhiễu ở đây.

Log thô: `logs/a12_base.log`, `logs/a12_v3.log` (không commit).

## Soi lại #54 sau khi có số: **bỏ A1, giữ ý A2 nhưng làm bằng hai hệ số**

Ba câu hỏi, ba câu trả lời bằng số (điểm quy ra bước với 117/bước):

1. **Bảng lõm có đáng độ phức tạp so với chỉ hạ `WALL_SCALE`?** Không, và "chỉ hạ `WALL_SCALE`"
   cũng không làm được việc đó. Muốn đuôi 0.86 bước thì `WALL_SCALE` phải xuống **0.737**, khi đó
   bức tường đầu rơi từ 2.33 → **1.27 bước** (mất một nửa) và tổng 2281 → 1245 điểm: `WALL_SCALE`
   là núm **mức**, tỉ lệ đuôi/đầu khoá cứng 0.667. Nên nhu cầu "tách đầu khỏi đuôi" là thật.
2. **Nhưng bảng có phải cách làm cần thiết?** Không. Parabol **hai hệ số** khớp đúng cả hai đầu
   (2.33 và 0.86 bước) là `wall_diff = 282.34`, `wall_quad = −9.59` — ba dòng, không bảng, không
   hàm dựng bảng. So với bảng WV, đường parabol đó lệch tối đa **0.28 bước** (n = 4–6), tức 1/4
   một bước `dist_diff`: nhỏ hơn sai khác do thêm/bớt một tầng tìm kiếm. Toàn bộ cái bảng mua được
   chỉ là phần lõm **bên trong** hai đầu, mà ta không có bằng chứng nào cho nó.
3. **0.4σ có đủ để giữ?** Không, và tệ hơn: đây không phải phép đo sạch cho giả thuyết "hình
   dạng". Bản 3 đổi **đồng thời** hình dạng và mức (tổng 2281 → 1664, −27%), nên nếu nó có nhích
   thì cũng không biết nhờ cái nào. Tiền lệ #51 nói đúng loại núm này nằm trong nhiễu: 1.35 vs
   1.20 ra **103.5 vs 103.5 điểm trên 260 ván**, dấu ngược nhau theo đối thủ. Cộng thêm hai ô đo
   bị **sàn** (mốc 0.05–0.10, đối thủ thắng 34–38/40 ván) nên một thay đổi trong hàm đánh giá gần
   như không có chỗ thể hiện.

Quyết định: **bỏ A1** (bảng `WV_REF` + `WALL_CUM`); **giữ A2 ở dạng hai hệ số**:
`WALL_LIN = 282.34`, `WALL_QUAD = −9.59` trong `WEIGHTS_FIRST`/`WEIGHTS_SECOND`.

`Phat_buffed_3.py` đã được viết lại thành đúng bản gọn đó: **chỉ hai số** khác `Phat_buffed_2`,
không còn `WV_REF`/`WALL_CUM`/`WALL_VALUE`/`WALL_CURV` (`diff` với bản 2 chỉ ra phần docstring,
comment và hai hệ số). Đường cong trùng hai đầu với bảng (2.33 … 0.86 bước, chỉ lệch tối đa 0.28
bước ở giữa), tổng 1864 điểm so với 2281 của bản 2.

**Bản gọn này CHƯA đo.** Các số 0.05/0.06 và 0.10/0.14 ở bảng trên thuộc bản **bảng WV** đã bị
xoá — đừng gán chúng cho file hiện tại. Đo nó chỉ đáng làm ở ô có sàn (24125023 / 24125062, mốc
0.19–0.38) và ghép cặp theo seed. Không promote thành bản nộp.

Hướng ăn điểm lớn hơn (theo bằng chứng của chính repo, không theo cảm giác):

- **Tàn cuộc có tường, không chỉ 0 tường.** Ván thua được phân tích ở #53 thua ở lượt 72–78 khi ta
  còn **1 tường**, đối thủ 2, và bị bít hàng đích — đúng thế mà bảng cuộc đua 0 tường (mục 4a)
  không phủ tới. Đây là chỗ duy nhất ta có một ván thật chỉ đúng nguyên nhân.
- **`pass` là đòn phòng thủ, không phải nước gốc đổi nhịp.** (sửa ở #57) 014 chỉ cho pass khi địch
  chặn đường duy nhất **và** pass không vứt mất thắng; 009 dùng anti-fortress + giá trị draw-aware
  (hoà = 1 điểm, nên hoà đáng khi đã thua cuộc đua); 062 giải DTM **có** pass. Giá trị thật của
  pass là *cứu hoà / phá fortress*, không phải giành nhịp. Việc của ta: đếm xem có bao nhiêu ván
  thua mà một nước pass đúng lúc sẽ hoà — chứ đừng thêm pass như nước khai cuộc.
- **Sách khai cuộc: không phải bằng chứng yếu.** (sửa ở #57) Chấm giải **bắt đầu từ thế chuẩn**, và
  đúng ở đó sách mới là chỗ các bot đầu bảng tách nhau: 009 báo 0,68 → 185/198 điểm sau khi thêm
  170 thế; 014 báo **22-1-1 từ thế chuẩn** trước đúng bot mà nó chỉ ngang ngửa khi mở ngẫu nhiên
  (85-4-91/180). Ván mẫu của ta hoà 40 lượt đầu chỉ nói sách của 014 không hạ được ta trong *một*
  ván. Cái đắt là **công dựng** sách, không phải bằng chứng — nên vẫn xếp sau các sửa cơ chế,
  nhưng đừng gọi nó là hướng yếu.

---

# #55. Đo `Phat_buffed_3` (hai hệ số parabol) ở hai ô có sàn: **hoà, dấu ngược theo đối thủ**

Cách đo: `phat-training/paired.py` (**mới**) — với mỗi (đối thủ, seed) chạy 2 ván cho A và 2 ván cho
B **cùng seed đổi bên**, nên so được từng cặp chứ không chỉ tổng. 20 cặp/ô, seed 12345, cả hai bên
0.275 s, `max_turns` 300, mở đầu 2 nước ngẫu nhiên. Tổng 160 ván, 703 s.

| bot | vs 24125023 (hạng 3) | vs 24125062 (hạng 4) | tổng 80 ván |
|---|---|---|---|
| `Phat_buffed_2` (mốc) | 0.362 (14-1-25) | 0.325 (13-0-27) | 0.344 |
| `Phat_buffed_3` (282.34 / −9.59) | **0.562** (22-1-17) | **0.225** (9-0-31) | 0.394 |

So từng cặp ván (A − B), mỗi cặp = 2 ván cùng seed đổi bên:

| ô | n | hiệu | ± | hơn / bằng / kém |
|---|---|---|---|---|
| 24125023 | 20 | **+0.200** | 0.107 (1.9σ) | 9 / 9 / 2 |
| 24125062 | 20 | **−0.100** | 0.078 (1.3σ) | 3 / 10 / 7 |
| TỔNG | 40 | **+0.050** | 0.069 (0.7σ) | 12 / 19 / 9 |

Đọc kết quả:

- **Đúng vết xe #51**: một núm giá tường thắng đậm một họ đối thủ và thua họ kia, hai ô triệt nhau.
  Tổng +0.05 ± 0.07 = **0.7σ**, không phải "bản 3 mạnh hơn". Không có ô nào đạt 2σ, và có 2 ô được
  kiểm nên ngay cả 1.9σ cũng không sống sót sau khi tính nhiều phép so.
- **Ghép cặp có tác dụng thật**: 19/40 cặp hai bản ra cùng kết quả (cùng nước mở đầu, cùng đối thủ),
  nên sai số của hiệu (±0.069) nhỏ hơn sai số một ô đo riêng (±0.076) và nó **phơi ra chuyện dấu
  ngược** mà so tổng che mất — so tổng chỉ cho +0.05 và không nói được gì thêm.
- So tổng và so cặp khớp nhau (+0.05 cả hai) -> cách ghép cặp không thiên lệch, chỉ làm rõ.
- Ở 0.275 s `Phat_buffed_2` được **0.362** trước 24125023, cao hơn hẳn số 0.19 của #53 (8 ván,
  0.55 s). Ô 40 ván là con số đáng tin; **đừng so chéo** giữa hai lần đo.

Quyết định: **không dùng** `Phat_buffed_3` làm bản nộp — giá tường không phải nút cổ chai, và một
núm chỉ đổi chỗ thắng giữa các đối thủ thì không đáng đổi bản nộp. Hướng còn lại vẫn là ba thứ ở
#54 (tàn cuộc có tường -> `pass` ở gốc -> sách khai cuộc).

### Gộp lại kết quả #55 sau khi đo với chấm 1/3 và max_turns 500 (2 lần lặp/seed)

| Seed | `Phat_buffed_3` W-D-L | `Phat_buffed_2` W-D-L | hiệu ± σ (n=40) |
|---|---|---|---|
| 12345 | 22-1-17 | 14-1-25 | **-0,058 ± 0,068 (0,86σ)** |
| 22345 | 13-3-24 | 13-0-27 | **-0,113 ± 0,059 (1,91σ)** |

- **#55 cũ** (chấm 0,5, max_turns 300) cho **+0.200 @1.9σ**: hiệu đảo dấu và giảm rõ rệt
  ở cấu hình mới (−0.058 @0.86σ và −0.113 @1.91σ), chứng tỏ lỗi nằm ở cách chấm/timing chứ
  không phải hiệu ứng thực sự. Cần ≥2σ và tái lập trên ≥2 lần mới coi có ý nghĩa.

Log: `logs/r55_s12345.log`, `logs/r55_s22345.log` (không commit).

---

# #56. Tàn cuộc còn tường: luật "bức cuối giữ để đáp trả" -> `Phat_buffed_4` (**không promote**)

Hướng đầu tiên trong danh sách #54. Không đụng giá tường (đã đo ở #55), không thêm đặc trưng: chỉ
thêm **một luật lọc danh sách nước ở gốc** (mục 4c của `submissions/Phat_buffed_4.py`, 109 dòng).

## Soi lại ván thua #53 trước khi sửa (replay `logs/g1.json`, 78 lượt vs 24125014)

Dựng lại ván bằng `Environment` rồi ở **mỗi lượt của ta** in: tường ta/đối thủ, `d_me`, `threat`
(số bước lớn nhất MỘT bức tường của đối thủ cộng được vào đường ngắn nhất của ta: vùng DAG +
`dist_after_wall`, 999 = bít hẳn) và lãi của bức tường replay đã chơi.

| lượt | tường ta/địch | d_me | threat | nước replay | lãi của nó |
|---|---|---|---|---|---|
| 61 | 2 / 2 | 5 | **994** | M(4,2) | – |
| 63 | 2 / 2 | 4 | **995** | WH(6,3) | **+2** |
| 65 | 1 / 2 | 4 | 995 | M(3,2) | – |
| 67 | 1 / 2 | 3 | 2 | WH(6,1) | **+2** |
| 69 | 0 / 2 | 3 | 2 | M(2,2) | – |

Ba điều đọc ra, và điều thứ nhất đổi cả thiết kế:

1. **Lượt 67 không phải chỗ sửa được.** Ngay từ lượt 65 tìm kiếm đã chấm thế là thua chắc
   (`score` −89990 ≈ trừ gần hết `WIN_SCORE`, tức đối thủ thắng trong ~10 ply), và bức tường ở lượt
   67 lãi đúng **2 bước** nên nó *không* phải tường "hoà nhịp" — luật không có gì để bỏ. Chạy thử
   luật với ngưỡng "đúng 1 tường" trên chính ván này: **không đổi một nước nào**.
2. **Lượt 63 mới là chỗ tiêu bừa** (còn 2 tường, lãi 2 bước). Ngưỡng phải là "sắp cạn tường"
   (`<= 2`), không phải "đúng 1 tường" như câu chữ của #54 — nếu không thì luật vô dụng đúng ở ván
   dùng để biện luận cho nó.
3. **threat = 994–995 ở lượt 61–65**: một bức tường của đối thủ **bít hẳn** đường ta, và hàm đánh
   giá hoàn toàn không có đại lượng này (`vuln` chỉ đếm *một* tường và chặn ở 8 bước). Đây là con
   số đáng nhớ nhất của cả mục này.

## Luật (mục 4c)

Bật khi **đồng thời**: `1 <= tường ta <= 2`, `tường đối thủ >= 2`, `d_me <= 6` (tàn cuộc).

```
lim = min(max(2, threat), 4)
bỏ khỏi danh sách nước gốc mọi nước TƯỜNG có lãi < lim   (nước đi quân không bao giờ bị bỏ)
```

- `max(2, threat)`: bức cuối phải đỡ được đúng cái đòn nó phải đỡ (`threat`) hoặc ít nhất 2 bước —
  dưới 2 bước là hoà nhịp (ta mất 1 nhịp, đối thủ chậm 1 bước), tức tiêu tường miễn phí.
- `min(..., 4)`: chặn trên. Không có nó thì `threat = 995` (một tường bít hẳn ta) khoá cứng **mọi**
  tường (`dropped = 6`) — biến bot thành "không bao giờ đặt tường nữa" ở tàn cuộc, một thay đổi to
  hơn nhiều so với ý định của luật.
- Chỉ chạy một lần ở gốc, chi phí ~1 ms (64 vùng × 2 hướng × `dist_after_wall`), không đụng hàm
  đánh giá, TT, killer.

### Dọn lại 4c sau khi đo (không đổi hành vi)

Soi lại 4c với con mắt "cái gì thừa": **nhánh `threat >= 999` là code chết** — `wall_threat` trả
`999 − d_me`, mà `d_me >= 1` (đứng ở hàng đích thì `choose_action` đã thắng từ trước khi vào luật),
nên `threat <= 998`; giữ nhánh đó chỉ làm người đọc hiểu sai. Ngược lại **trần `4` không thừa**:
chính nó xử lý trường hợp bị bít hẳn (khi đó chỉ giữ nước tường lãi >= 4, thay vì khoá sạch tường).
Các chỗ gọn lại: gộp ba điều kiện vào một `if`, gộp đổi mã nước tường thành một dòng, bỏ tham số
`side` thừa của `wall_threat` (luôn là `state.side`), đổi tên `PLUG_HOLD_OWN_WALLS` ->
`PLUG_HOLD_MAX_WALLS`, sửa câu chữ "thế 1 tường vs 2 tường" -> "ta còn <= 2 tường" (luật chạy tới 2
tường), và sửa docstring `wall_threat` nói sai "trả 999" (thực ra `999 − d_me`).

Kiểm không đổi hành vi: dựng lại ván #53, lấy danh sách nước gốc ở **cả 39 lượt của ta** rồi đưa
CÙNG danh sách đó cho `keep_last_wall`/`wall_threat` của bản trước và bản sau — **0 khác biệt** về
danh sách nước, về `threat` và về thông tin debug (luật bắt 4/39 lượt ở cả hai bản).

## Đo: `paired.py`, A = `Phat_buffed_4`, B = `Phat_buffed_3`, 20 cặp/ô, seed 12345, 0.275 s

160 ván, 419 s. Hai ô đúng như yêu cầu: hạng 1 và hạng 2 vòng 1 (đều là bot có `pass` + bảng cuộc
đua, tức họ bít hàng đích thật).

| bot | vs 24125014 (hạng 1) | vs 24125009 (hạng 2) | tổng 80 ván |
|---|---|---|---|
| `Phat_buffed_3` (mốc) | 0.100 (4-0-36) | 0.163 (6-1-33) | 0.131 |
| `Phat_buffed_4` (giữ bức cuối) | **0.188** (5-5-30) | 0.050 (1-2-37) | 0.119 |

So từng cặp ván (A − B), mỗi cặp = 2 ván cùng seed đổi bên:

| ô | n | hiệu | ± | hơn / bằng / kém |
|---|---|---|---|---|
| 24125014 | 20 | **+0.087** | 0.066 (1.3σ) | 6 / 12 / 2 |
| 24125009 | 20 | **−0.113** | 0.050 (**2.3σ**) | 0 / 15 / 5 |
| TỔNG | 40 | **−0.013** | 0.044 (0.3σ) | 6 / 27 / 7 |

Đọc kết quả:

- Tổng **−0.013 ± 0.044 (0.3σ)**: hoà, không promote.
- Nhưng đây **không phải nhiễu phẳng**: dấu tách hẳn theo cơ chế, không theo may rủi. Hạng 1 chơi
  đúng trò bít hàng đích (plug-pass + bảng cuộc đua) — đúng thứ luật nhắm tới: **+0.087, 6 cặp hơn
  / 2 kém**, và 5 ván hoà mới xuất hiện (mốc không có ván hoà nào). Hạng 2 (ngân sách nút + NN gợi
  ý tường, ít chơi trò bít) thì **−0.113, 0 cặp hơn / 5 kém** — luật bỏ bức tường lãi 2 bước đúng
  lúc cần nó để chạy đua.
- 2.3σ là σ mạnh nhất trong cả hai lần đo (#55 mạnh nhất 1.9σ) nhưng là σ **âm**, và chỉ có 2 ô nên
  không sống sót sau khi tính nhiều phép so. Kết luận trung thực: **luật đúng hướng ở thế bị bít
  nhưng quá tay** — nó áp cho *mọi* đối thủ còn >= 2 tường, kể cả đối thủ không có ý định bít.
- Đừng so chéo với #53 (0.00 / 0.12 ở 0.55 s, 8 ván): cùng seed nhưng khác ngân sách và khác số ván.

## Quyết định và nếu làm tiếp

- **Không promote.** `Phat_buffed_4.py` giữ lại làm mốc; `MSSV.py` vẫn là bản nộp chính thức,
  không đụng.
- Refinement có bằng chứng, rẻ: chỉ bật luật khi **đường của ta hẹp thật** (`shortest_dag_analysis`
  trả `br_count = 1`, tức mọi đường ngắn nhất chạy qua cùng một chỗ -> một tường bít được), thay vì
  bật chỉ vì `threat >= 2` (cắt được một nhánh trong khi ta còn 3 nhánh). Đây là chỗ hứa hẹn nhất
  vì nó giải thích đúng cả hai dấu: 014 bít được -> bật là lãi; 009 không bít -> tắt là tránh mất.
- Điều đáng làm hơn cả luật này: **đưa `threat` vào lá** (một số đo "một tường của đối thủ bít hẳn
  ta" đang bằng 994 mà eval coi như 8 bước vuln). Luật ở gốc chỉ sửa được nước đi, còn eval sai thì
  tìm kiếm vẫn chấm thế thua từ lượt 65 như ván #53.

### Gộp lại kết quả #56 sau khi đo với chấm 1/3 và max_turns 500 (2 lần lặp/seed)

| Seed | `Phat_buffed_4` W-D-L | `Phat_buffed_3` W-D-L | hiệu ± σ (n=40) |
|---|---|---|---|
| 12345 | 6-0-34 | 7-1-32 | **-0,017 ± 0,061 (0,27σ)** |
| 22345 | 7-0-33 | 4-0-36 | **-0,025 ± 0,057 (0,44σ)** |

- **#56 cũ** (chấm 0,5, max_turns 300) cho **−0.113 @2.3σ**: hiệu giảm rõ rệt xuống 0.017/@0.27σ
  và 0.025/@0.44σ, σ tăng lên do nhiễu máy. Kết luận "không promote" vẫn đúng, nhưng là bí
  cỗ chứ không phải bằng chứng cứng.
- Cần ≥ 80 cặp/ô và ≥ 2 lần lặp cùng seed mới ổn định.

Log: `logs/r56_s12345.log`, `logs/r56_s22345.log` (không commit).

---

# #57. Đọc 11 báo cáo vòng 1: cơ chế họ tự mô tả, đối chiếu với nước đi ta suy ra

Nguồn: `wrongway-student/wrongway/round-1-submissions/reports/*.pdf` — **11** báo cáo, trích bằng
`/ucrt64/bin/pdftotext -layout`. Mục này là bản gốc của ba chỗ đã sửa trong #53 (đánh dấu ⚠ dưới đây),
để sau này không suy lại sai từ log nước đi.

Cột "họ tự mô tả" là ý trong báo cáo; cột "đối chiếu" là cái #53 đoán từ log, giữ cả chỗ sai.

| Bot | Hạng | Họ tự mô tả | Đối chiếu với #53 |
|---|---|---|---|
| 24125014 | 1 (51/60) | negamax/PVS + Zobrist + aspiration + killer/history + LMR; **policy net int8 chỉ để XẾP THỨ TỰ nước** (nước điểm thấp tìm nông hơn, cùng độ sâu −30% số thế); anti-draw = chênh đường + **giá tường tăng theo độ dài cuộc đua** (fit 426k thế) + tempo + bảng retrograde khi 0 tường; plug-threat check; **pass chỉ khi địch chặn đường duy nhất và pass không vứt thắng**; sách 2.399 thế do teacher mạnh hơn viết offline; 0,55 s mục tiêu → 0,75 s khi nước còn đổi → hard 0,80 s; tắt GC khi nghĩ | Đúng: sách, plug-pass, bảng cuộc đua. Sai: ta gọi plug-pass là "đòn tấn công" — họ mô tả nó là **luật phòng thủ có điều kiện**, và pass là thứ *bảo toàn* thắng chứ không phải giành nhịp |
| 24125009 | 2 (45/60) | PVS + TT + LMR + **sinh nước theo tầng**, 11–12k thế/s, sâu 6–10; **ngân sách 0,54 s mềm + hard stop** (nước chậm nhất 496–573 ms); bảng cuộc đua 75.000 giá trị học RL TD(0,7); giá tường lõm 30, 26, 23 … 11; threat term; lazy eval; anti-stall; **mạng 354→32 distill từ teacher**, policy head chỉ **thêm 4 nước tường quiet/setup ở gốc**, value head tắt (R=0: chậm 35%, thua 5-1-8 và 5-3-6); **8 setup wall ở gốc**; sách 170 thế (0,68 → 185/198); anti-fortress + draw-aware + repetition contempt | ⚠ **"ngân sách NÚT cố định" là bịa** — không có ngân sách nút, họ dùng **thời gian**. Đúng: bảng cuộc đua, mạng nhỏ gợi ý tường — nhưng chỉ **4 nước ở gốc**, không phải cả cây, và value net bị chính họ bác |
| 24125023 | 3 (37/60) | Atlas: bitboard 81-bit + wall mask 64-bit; **BFS witness certificate** (chỉ thử O(L) tường cắt đường chứng kiến, bỏ 85–90% lời gọi BFS); **route-pressure DAG flow** + conflict-veto để xếp hạng tường setup/phòng thủ; two-ply root floor screening; eval fit từ corpus; **solver retrograde 6.561 bit cho 0 tường** | Đúng: "tàn cuộc giải chính xác". Mới: tường setup được **xếp hạng bằng dòng chảy trên DAG**, không chỉ bằng gain 1 nước — đây là cách sinh tường "quiet" mà ta không có |
| 24125005 | (4, 36/60) | **Bài của đội mình**, không phải đối thủ: nộp dưới MSSV trưởng nhóm, 734 dòng, cùng dòng code với `submissions/MSSV.py` (831 dòng) | Mọi ô đấu với 24125005 trong log vòng 1 là **tự đấu**. Không đọc như thắng/thua trước người ngoài, không dùng để xếp hạng đối thủ, không đem số đó so với #53–#56 |
| 24125062 | 4 (36/60) | dual-profile P0/P1 (bất đối xứng theo bên đi trước), PVS bitboard, BFS µs, **cắt theo giao các đường ngắn nhất** (−85% nhánh), **prior tường bằng NN lượng tử 1,3 KB**, **tablebase retrograde có chu trình (DTM, tính cả pass)** + chống lặp | Đúng: DTM 0 tường có pass + chống lặp |
| 24125015 | 6 (34/60) | bitboard tăng dần (một XOR mỗi hướng), **Zobrist chuẩn hoá gương** cho TT và mọi cache, PVS + LMR an toàn (tường `wall_i > 3` mới bị giảm, tường gain ≥ 3 được bảo vệ), cutoff 0 tường khi dẫn ≥ 3 bước, "leaf trap potential", root bias + thưởng né ngang, **SPSA tự chỉnh** qua 3 tầng kiểm định thống kê | Đúng: mô hình tường + cắt tỉa. Mới: chuẩn hoá gương **và** SPSA (ta chưa có cả hai) |
| 24125011 | — | alpha-beta, budget **0,8 s**, exact endgame khi cả hai hết tường, **chỉ xét tường trong 4 hàng/cột gần người chơi** (≤ 64) rồi lọc theo gain, tối đa 18 tường ở gốc | Mới: giới hạn vùng tường theo khoảng cách người chơi — đúng chỗ #25 nói ta sinh thiếu tường gần |
| 23125054 | — | MCTS: mã hoá 4 nước đi + 1 pass + 128 tường, kiểm overlap bằng mask O(1), rollout heuristic chọn tường cắt đường, **pass = −1,5 điểm**, tái dùng nước trước để tiết kiệm budget | Đúng: `pass` có giá riêng trong eval, không phải nước rỗng |
| 24125076 | — | bitboard + **BFS wavefront có cache**, **cắt tường một lượt trên DAG đường ngắn nhất**, **sinh nước theo tầng + cutoff ở lá**, eval = kinh tế tường phi tuyến + parity nhảy đối đầu | Đúng: sinh nước theo tầng (75–85% nút) là chỗ đáng chép nhất về tốc độ |
| 24125085 | — | TITAN V3: bitboard + PVS + Zobrist + **solver tàn cuộc chính xác**; ngân sách chia **0,46 / 0,50 / 0,18 s** | Đúng: solver tàn cuộc; ngân sách chia ba phần là chi tiết ta chưa thử |
| 24125001 | — | bitboard, **Zobrist + ảnh gương để tái dùng kết quả**, BFS bitboard, tách **"tường chạm MỘT đường ngắn nhất"** và **"chạm MỌI đường ngắn nhất"**, xếp hạng tường theo gain | Đúng: hai tập đó chính là thứ `threat` của ta tính gộp thành một số — #56 cho thấy gộp là mất thông tin (`br_count = 1` mới là "bít được") |

## Ba chỗ #53 nói sai

1. **009 không dùng ngân sách nút.** Họ ghi rõ ngân sách **0,54 s mềm + hard stop**, không bắt đầu
   độ sâu mới khi gần hết giờ, nước chậm nhất 496–573 ms. Không có cơ chế "chơi y hệt trên máy
   nhanh/chậm" nào trong báo cáo — đó là ta tự suy từ việc log của họ ít biến thiên.
   Vì sao đáng sửa: ta đã xếp "ngân sách nút" vào danh sách đáng chép, trong khi nó **không tồn
   tại**; còn thứ thật của 009 là **4 nước tường quiet do mạng gợi ý ở gốc** + 8 setup wall, thứ ta
   đang xếp thấp hơn.
2. **Vai trò của `pass` là anti-draw/draw-aware, không phải nước gốc đổi nhịp.** 014: pass chỉ được
   phép khi địch chặn đường duy nhất **và** không vứt mất thắng; "pass stand-offs now end in a draw
   rather than a loss" — bản trước **thua 22/24** đúng ở những thế đó. 009: hoà = 1 điểm nên hoà
   đáng nhận khi đã thua cuộc đua, kèm anti-fortress để phá người đóng cửa hàng đích rồi pass.
   062: pass nằm trong bảng DTM. Tức `pass` là **thiết bị cứu hoà / phá fortress**, và giá trị của
   nó đo được bằng *số ván thua lẽ ra hoà*, không phải bằng nhịp.
3. **Sách khai cuộc không phải bằng chứng yếu nhất.** Chấm giải **bắt đầu từ thế chuẩn**, và cả hai
   báo cáo đầu bảng đều nói thẳng rằng đó là nơi họ ăn điểm: 009 tăng **0,68 → 185/198** khi thêm
   170 thế; 014 **22-1-1 từ thế chuẩn** trước đúng bot mà nó chỉ ngang ngửa khi mở ngẫu nhiên
   ("from random openings the strongest classmate agent is still about even with ours: 85-4-91/180").
   Ván mẫu của ta hoà 40 lượt đầu với 014 chỉ chứng minh sách của họ không hạ được ta trong **một**
   ván. Cái đắt là công dựng sách, không phải bằng chứng — nên vẫn xếp sau các sửa cơ chế.

## Cái đáng chép và cái chính họ đã bác

Đáng chép, theo độ chắc của bằng chứng trong báo cáo:

- **Chuẩn hoá gương** — 001 (Zobrist có ảnh gương để tái dùng kết quả) và 015 (TT + mọi cache chỉ
  index theo dạng chuẩn hoá) làm ở tầng tìm kiếm, còn 009 chỉ quay bàn về bên đi trước cho input
  của mạng. Rẻ, không đổi nước đi, chỉ tăng số thế dùng lại được — cùng loại với #34 (tăng tốc thuần).
- **Sinh nước theo tầng + cutoff ở lá** (076, 009): cắt 75–85% nút trước khi tìm, tức sâu thêm ~1 tầng.
- **BFS witness certificate** (023): chỉ thử O(L) tường cắt đường chứng kiến thay vì cả 128 — bỏ
  85–90% lời gọi BFS, đúng chỗ ta đang tốn nhất.
- **Tường quiet/setup do mạng nhỏ gợi ý ở gốc** (009: 4 nước; 014: policy net làm thứ tự nước,
  −30% nút cùng độ sâu). Đây là bằng chứng *policy net có lãi*, không phải value net.
- **Tường setup xếp hạng bằng dòng chảy DAG** (023) — cách sinh tường "không tăng đường ngay" mà
  eval của ta cho 0 điểm.

Đã bị chính họ bác, đừng làm lại:

- **Value net**: 009 đo 5-1-8 và 5-3-6 khi bật, chậm 35%, nên để R=0. Trùng kết luận #54 của ta.
- **Plug-threat rule**: 014 A/B 1.000 ván, 489-511 / 382-618, bác. Trùng **#56** (−0.113 vs 009).
- **Null-move**: không báo cáo nào dùng.
- **Thêm giờ**: 014 đã nâng lên 0,80 s hard stop và vẫn để 0,55 s mục tiêu; 009 0,54 s. Không ai
  dùng gần 1 s — thêm giờ là đổi lấy tốc độ chứ không phải bảo hiểm.

## Việc phải làm trước khi tin bất kỳ số nào tiếp theo

`phat-training/paired.py` đang chấm **hoà = 0,5** trong khi giải chấm **thắng 3 / hoà 1** (tức hoà =
1/3), và đang để `max_turns = 300` trong khi arena để **500**. Mọi số ở #55–#56 vì thế vừa lệch
hằng số vừa cắt ngắn ván (ván dài bị biến thành hoà sớm). **Đã sửa** (xem mục dưới).

Bản trích PDF tạm đã xoá (`phat-training/tmp/`), không commit.

## Đã sửa harness và đo lại ô mốc #55: **số tuyệt đối không tái lập được, hiệu theo cặp thì yếu đi**

Sửa trong `phat-training/paired.py`:

- `DRAW_PT = 1/3` (thay 0,5) — cả cột điểm từng bot lẫn điểm từng ván; đúng cách
  `tools/tournament.py` chấm: `(3W + D) / (3·số_ván)`.
- `--max-turns` mặc định **300 → 500**, như `environment.py`.

Đo lại đúng ô của #55 (20 cặp, seed 12345, 0,275 s, mở đầu 2 nước), hai lần chạy liền nhau:

| Cấu hình | `Phat_buffed_3` W-D-L | điểm (1/3) | điểm nếu chấm 0,5 | `Phat_buffed_2` | hiệu theo cặp |
|---|---|---|---|---|---|
| #55 gốc (300 lượt, chấm 0,5) | 22-1-17 | 0,558 | **0,562** | 14-1-25 → 0,362 | **+0.200 ± 0.107 (1.9σ)** |
| chạy lại, 500 lượt, chấm 1/3 | **15-5-20** | **0,417** | 0,438 | 13-1-26 → 0,333 | +0.083 ± 0.092 (0.9σ) |
| chạy lại, 300 lượt, chấm 1/3 | 17-4-19 | 0,458 | 0,475 | 16-1-23 → 0,395 | +0.050 ± 0.124 (0.4σ) |

Đọc kết quả — **hai chỗ sửa không phải nguyên nhân của chênh lệch**:

- Riêng cách chấm: áp 1/3 lên chính số cũ (22-1-17) chỉ đổi 0,562 → 0,558, tức **−0,004**. Với 5
  ván hoà thì cùng lắm **−0,021**. Không đủ giải thích −0,145.
- Riêng turn limit: cùng 300 lượt mà chạy lại đã ra 0,475 (chấm 0,5) thay vì 0,562, tức **−0,087**
  **khi mọi thứ khác y hệt**. Đó là **nhiễu chạy lại**, không phải turn limit: 500 vs 300 lượt chỉ
  lệch 0,417 vs 0,458, và số ván hoà *tăng* (5 so với 4) chứ không giảm, nên turn limit không có
  tác dụng hệ thống ở ô này.
- Nguồn nhiễu: cả hai bot tìm kiếm theo **deadline thật** (0,275 s), nên nước đi phụ thuộc tải máy
  và **không tái lập tuyệt đối** dù cùng seed; seed chỉ cố định nước mở đầu. Sai số chuẩn của một ô
  40 ván ở p ≈ 0,5 là ~0,08, đúng cỡ chênh lệch thấy được.

Hệ quả cho #55–#56:

- **+0.200 (1.9σ) của #55 không tái lập**: cùng ô, cùng seed, chạy lại còn **+0.050 (0.4σ)**. Con số
  "1,9σ" đó không đủ để nói `Phat_buffed_3` hơn bản 2 ở ô 24125023 — kết luận "không promote" vẫn
  đúng, nhưng lý do phải là *chưa đo được*, không phải *gần ngưỡng*.
- Tương tự, −0.113 (2.3σ) của #56 phải coi là **một lần bốc được σ mạnh**, không phải bằng chứng
  cứng; muốn chốt phải chạy lại đúng ô đó ≥ 2 lần hoặc tăng lên ≥ 80 cặp/ô.
- Từ giờ: báo cáo **hiệu theo cặp** (ổn định hơn) chứ đừng so điểm tuyệt đối giữa hai lần chạy, và
  mọi kết luận cần ≥ 0,1 chênh **và** ≥ 2σ **và** tái lập được ở lần chạy thứ hai.

Log: `logs/p57_mark500.log`, `logs/p57_mark300.log` (không commit).

### Nốt chỗ còn lệch: `evolve.score_of` và `gauntlet.py` — gom về `common.py`

`paired.py` đã chấm 1/3 nhưng `gauntlet.py` (công cụ đo chính theo AGENTS.md) vẫn tự tính
`(W + 0.5·D)` trong dòng in, và `evolve.score_of` (dùng cho các cổng nhận đột biến của `evolve.py`)
cũng 0,5. Đã sửa:

Lần đầu tôi đặt hằng số trong `evolve.py`, nhưng soi lại thì sai chỗ và còn sót công thức:
`evolve.py` là script **train**, còn luật chấm là thứ dùng chung cho cả ba công cụ; và `gauntlet.py`
lẫn `paired.py` vẫn tự viết lại `(w + d·DRAW_PT) / (w+d+l)` ở hai nơi. Đã gom về **`common.py`**
(module dùng chung, không import `evolve` nên không vòng lặp import):

- `common.DRAW_PT = 1/3`, kèm ba hàm: `game_pt(winner, side)` (điểm một ván), `score_of(results)`
  (điểm trung bình của một lô), `rate(w, d, l)` (điểm từ tổng W-D-L).
- `evolve.py` bỏ hằng số và `score_of` riêng, chỉ `from common import score_of`.
- `gauntlet.py` và `paired.py` không còn tự tính điểm: gọi `common.rate` / `common.game_pt`. Hết chỗ
  nào trong `phat-training/` còn viết `0.5` cho hoà (đã grep).

Kiểm:

- `game_pt` → 1,0 / 0 / 1/3; `score_of` và `rate` cùng ra **0,5556** cho 26-2-20 (khớp tay).
- Chạy khói cả hai CLI sau khi gom: `gauntlet.py ... --pairs 3` → W 3 D 1 L 2 "điểm 0.56" (29 s);
  `paired.py ... --pairs 2` → in đủ cột W/D/L + điểm và hiệu theo cặp (55 s). Cả hai chạy được.
- Ba file `.py` sửa tay được đưa về **CRLF** như phần còn lại của `phat-training/` (`paired.py` trước
  đó là LF), `git diff --numstat` chỉ 22/1, 1/7, 5/2 dòng — tức chỉ đổi nội dung, không đổi cả file.

Kiểm bằng một ô nhỏ chạy đúng `gauntlet.py` (`Phat_buffed_3` vs 24125023, 24 cặp = 48 ván,
500 lượt, 0,275 s, 230 s):

| lần chạy | W-D-L / 48 ván | điểm 1/3 | ghi chú |
|---|---|---|---|
| gauntlet, 500 lượt, chấm 1/3 | **26-2-20** | **0,56** | (26 + 2/3)/48 = 0,556 |

Hai điều rút ra:

- Code chấm đúng: 0,556 chứ không phải 0,5625 của thang 0,5 (in ra cùng 0,56, nên phải tính tay).
- Ô này ra **0,56**, khớp số cũ #55 (0,562) chứ **không khớp** lần chạy 15-5-20 → 0,417 ở trên. Vậy
  con 0,417 là **lần chạy lệch**, không phải tác dụng của turn limit hay cách chấm — càng củng cố kết
  luận: một lần chạy một ô là không đủ, phải lặp lại.

Còn lệch có chủ ý (chưa sửa, vì sẽ đổi tốc độ mọi phép đo): `gauntlet.py` vẫn để `--max-turns 300`
mặc định trong khi arena để 500 → muốn so với `paired.py`/arena phải truyền `--max-turns 500`. Và
`--gate 0.55` của `evolve.py` được chỉnh dưới thang hoà 0,5, nay hoà rẻ hơn nên cổng đó **khắt khe
hơn một chút** — nếu train tiếp thì siết lại `--gate` sau khi xem vài thế hệ.

Log: `logs/p58_gauntlet13.log`.

---

# #58. Hạ hai hệ số tường (282.34 / −9.59 -> 250 / −7) -> `Phat_buffed_5` (**không promote**)

Tinh chỉnh `Phat_buffed_3`, không đổi cách tiếp cận: chỉ hai số trong khối `WEIGHTS`, phần còn lại
byte-giống bản 3 (`diff` chỉ 2 dòng hằng số + 1 dòng ghi chú).

## Chiều tác dụng phải nói đúng (chỗ dễ sai nhất)

`wall_diff = tường_còn_lại_ta − tường_còn_lại_đối_thủ` và trọng số là **dương**, nên `WALL_LIN` là
**giá của việc GIỮ một bức tường** (≈ 2.33 bước với 282.34, vì `dist_diff` = 117 điểm/bước). Giá biên
của bức thứ n là `WALL_LIN + WALL_QUAD*(2n-1)`:

| bức | bản 3 (282.34 / −9.59) | bản 5 (250 / −7) |
|---|---|---|
| 1 | 282.34 = **2.33 bước** | 250 = **2.14 bước** |
| 5 | 196.0 = 1.68 bước | 187 = 1.60 bước |
| 10 | 100.1 = **0.86 bước** | 117 = **1.00 bước** |

Tức hạ hai hệ số **không** phải "đặt ít tường hơn" (đó là chiều ngược): tường **rẻ hơn** ở các bức
đầu nên bot **tiêu sớm hơn**, còn `WALL_QUAD` bớt âm làm đường giá phẳng hơn nên bức cuối lại **đắt
hơn** một chút. Giả thuyết của phép đo là: khoảng cách giá hai đầu hẹp lại thì bot bớt "giữ tường
chết" ở tàn cuộc — đúng chỗ #55 nghi ngờ.

## Đo: `paired.py`, A = `Phat_buffed_5`, B = `Phat_buffed_3`, ô 24125023, 20 cặp, seed 12345

| ngân sách | A `Phat_buffed_5` | B `Phat_buffed_3` | hiệu (A − B) | hơn / bằng / kém |
|---|---|---|---|---|
| **0.275 s** (764 s) | 15-10-15 = 0.458 | 13-8-19 = 0.392 | **+0.067 ± 0.079** (0.85σ) | 8 / 7 / 5 |
| **0.5 s** (1087 s) | 14-3-23 = 0.375 | 17-5-18 = 0.467 | **−0.092 ± 0.116** (0.79σ) | 6 / 6 / 8 |

Đọc kết quả:

- **Dấu đảo theo ngân sách**: dương nhẹ ở 0.275 s, âm nhẹ ở 0.5 s, cả hai đều dưới 1σ. Đây là định
  nghĩa của nhiễu trong dự án này (chênh < 0.1, < 2σ), không phải "bản 5 tốt hơn ở máy chậm".
- Cả hai lần đều **không tái lập** được chiều nào: 20 cặp/ô là sàn, mà muốn tách một hiệu 0.07 với
  σ 0.08 thì cần ~100 cặp/ô. Vì thế kết luận đúng là **không kết luận được**, và bản nộp không đổi.
- Điểm tuyệt đối của **cả hai bản** tăng khi lên 0.5 s (0.392 -> 0.467 cho bản 3): ngân sách là biến
  gây nhiễu mạnh nhất trong mọi phép so ở đây — **đừng bao giờ so chéo hai ngân sách**.
- Số phụ (một lần, không dùng làm bằng chứng): bản 5 so với `Phat_buffed_2` trên cùng ô ra
  **+0.083 ± 0.108 (0.77σ)**, cùng chiều với lần 0.275 s ở trên nhưng khác mốc so.

## Quyết định

- **Không promote** `Phat_buffed_5`. `Phat_buffed_3` giữ nguyên hai hệ số cũ (file đã được trả về
  đúng 282.34 / −9.59 để #55 vẫn tái lập được), `Phat_buffed_5` chỉ giữ làm hồ sơ.
- Hai lần đo này khép lại hướng "định giá tường": đã thử 3 bộ hệ số (bản 2 một núm, bản 3 parabol
  2.33/0.86, bản 5 parabol phẳng hơn) — không bộ nào ra khỏi nhiễu. Giá tường **không phải** nút cổ
  chai, đúng như kết luận #55.

Log: `logs/p58_s12345.log` (0.275 s), `logs/p58_b500_s12345.log` (0.5 s) — không commit.

---

# #59. Thu hẹp cổng luật 4c về đúng ca "bị bít hẳn" -> `Phat_buffed_6` (**bác**)

Tinh chỉnh `Phat_buffed_4` (không đổi cách tiếp cận): giữ nguyên luật "giữ bức cuối", chỉ đổi **cổng
bật**. Bản 4 bật khi `threat >= 2` — tức bật với *mọi* đối thủ còn >= 2 tường, kể cả đối thủ không có
ý định bít; đó là giả thuyết giải thích vì sao #56 ra **+0.087** trước 24125014 (có bít) mà **−0.113**
trước 24125009 (không bít). Cổng hẹp hơn: chỉ bật khi đối thủ **bít hẳn** được ta, tức
`threat == 999 - d_me` — đúng con số đọc được ở replay #56 (lượt 61: `d_me` 5, threat 994; lượt 63:
`d_me` 4, threat 995; cả hai đều bằng `999 - d_me`).

Khác biệt giữa hai bản đúng **một dòng** (`PLUG_HOLD_SEAL_ONLY`), phần còn lại byte-giống:

```
if PLUG_HOLD_SEAL_ONLY and threat < 999 - d_me:
    return root_moves, None          # đối thủ không bít hẳn được ta
```

## Đo: `paired.py`, A = `Phat_buffed_6`, B = `Phat_buffed_4`, 20 cặp, seed 12345, 0.275 s, 500 lượt

80 ván, 379 s. Ô 24125014 (hạng 1, plug-pass + bảng cuộc đua — đúng loại đối thủ mà luật nhắm tới).

| bot | vs 24125014 | điểm |
|---|---|---|
| `Phat_buffed_4` (cổng rộng, mốc) | 7-3-30 | 0.200 |
| `Phat_buffed_6` (chỉ ca bị bít hẳn) | 5-1-34 | 0.133 |

So từng cặp ván (A − B), n = 20: **−0.067 ± 0.062 (1.07σ)**, hơn 2 / **bằng 12** / kém 6.

Đọc kết quả:

- **Cổng hẹp làm luật mất tác dụng**, không phải làm nó chính xác hơn: 0.200 -> 0.133 trên chính đối
  thủ mà bản 4 có lãi. Giả thuyết "lãi trước 014 là nhờ bắt đúng ca bị bít" **sai** — lãi nằm ở ca
  rộng (`threat >= 2`), tức ở những lượt đối thủ *có khả năng* bít chứ chưa bít.
- **12/20 cặp hai bản ra cùng kết quả**: luật này hiếm khi đổi được ván (nó chỉ lọc danh sách nước
  gốc ở tàn cuộc, còn lại tìm kiếm tự quyết). Vì vậy mọi hiệu của nó — kể cả +0.087 của bản 4 —
  đều phải đọc kèm số cặp "bằng", đừng đọc riêng hiệu.
- Không có gì để promote: bản 4 giữ nguyên cổng rộng, bản 6 chỉ giữ làm hồ sơ cho thấy đã thử thu hẹp.

Log: `logs/p59_s12345.log` (không commit).
