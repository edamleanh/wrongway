---
name: wrongway-rules
description: Luật chơi Wrongway, định dạng observation/action, mô hình tường và gợi ý chiến thuật (BFS, minimax/alpha-beta, đánh giá tường). Dùng khi viết hoặc sửa chiến thuật trong submissions/MSSV.py, hoặc khi cần tra luật.
---

# Luật Wrongway (tóm tắt)

Nguồn đầy đủ: `wrongway-student/wrongway/STUDENT_GUIDE.md`, `PROBLEM_STATEMENT.md`; mã luật thật: `wrongway-student/wrongway/environment.py`.

- Bàn 9x9, ô `[row, col]`, row tăng xuống dưới. P0 xuất phát `[8,4]`, đích hàng 0, **đi trước**. P1 xuất phát `[0,4]`, đích hàng 8.
- Mỗi lượt đúng một hành động: đi 1 ô (có nhảy qua đối thủ; chỉ đi chéo khi ô sau đối thủ bị chặn), đặt tường, hoặc pass.
- 10 tường/người. `H (r,c)` chặn `[r,c]-[r+1,c]` và `[r,c+1]-[r+1,c+1]`; `V (r,c)` chặn `[r,c]-[r,c+1]` và `[r+1,c]-[r+1,c+1]`. Neo 0..7. Không chồng/cắt nhau, không được bít hết đường của ai.
- Hòa sau 500 lượt đơn (tính cả hai bên). Thắng 3, hòa 1, thua 0.

## Observation

`board_size, player_id, current_player, turn, max_turns, you{position, goal_row, walls_remaining}, opponent{...}, walls{horizontal:[[r,c]], vertical:[[r,c]]}, legal_moves:[[r,c]], legal_walls:[{orientation,row,col}], distances (9x9, tới đích của mình, bỏ qua quân, -1 = không tới), done, winner`.

## Action

`{"type":"move","to":[r,c]}` | `{"type":"wall","orientation":"H","row":r,"col":c}` | `{"type":"pass"}`.
Phải chọn trong `legal_moves`/`legal_walls` (nhớ thêm `"type":"wall"`).

## Hướng chiến thuật (tăng dần độ khó)

1. **Đi tham lam** theo `distances`, tie-break về giữa bàn (đã có trong MSSV.py).
2. **Đặt tường có điều kiện**: chỉ khi đối thủ gần đích hơn mình; chọn tường tối đa `(Δ đường đối thủ) − (Δ đường mình)`. Ưu tiên tường sát trước mặt đối thủ.
3. **Tiết kiệm tường**: đừng xài hết sớm; giữ vài tường cho lúc đối thủ còn 1-3 bước.
4. **Lọc ứng viên tường**: chỉ xét tường chạm các cạnh nằm trên một đường ngắn nhất của đối thủ (giảm từ ~128 xuống ~10-20), giúp tìm kiếm sâu hơn.
5. **Minimax / alpha-beta** với iterative deepening và deadline: eval = `dist_đối_thủ − dist_mình` (+ trọng số số tường còn lại, + bonus lượt đi). Nước đi lấy từ luật nhảy/chéo (cần tự cài lại mô phỏng trong file, vì không được import Arena — có thể tham khảo `environment.py` để cài cho đúng).
6. Kiểm tra trường hợp hai quân đứng cạnh nhau: `distances` bỏ qua quân, nên dùng `legal_moves` cho nước thật.

Sau mỗi thay đổi: chạy skill `tournament` để so sánh và `check-submission` trước khi nộp. Nhớ: mỗi nước ≤ 1 s trên Colab → dùng `time.perf_counter()` và cắt tìm kiếm theo deadline.
