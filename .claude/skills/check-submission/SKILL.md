---
name: check-submission
description: Kiểm tra bài nộp Wrongway trước khi lên Moodle - tên file MSSV, chỉ thư viện chuẩn, có choose_action, đấu cả hai bên không timeout/lỗi, thời gian mỗi nước. Dùng khi người dùng chuẩn bị nộp bài hoặc hỏi "nộp được chưa".
---

# Kiểm tra trước khi nộp

1. Chạy unit test: `python -m unittest discover tests` (đổi file bot bằng biến môi trường `BOT=<đường dẫn>`).
2. Chạy checker:
   ```
   python tools/check_submission.py wrongway-student/wrongway/submissions/<MSSV>.py
   ```
3. Tự rà thêm những điều checker không bắt được:
   - File không import gì từ `arena`, `environment`, hay file cục bộ khác; không đọc/ghi file; không `print` thừa (không cấm nhưng nên bỏ).
   - Không làm việc nặng ở mức module (import ≤ 5 s).
   - Mọi vòng tìm kiếm có kiểm tra deadline; `TIME_BUDGET` để dư cho Colab (máy chấm chậm hơn, nên giữ thời gian đo được trên máy ≤ ~0.3-0.5 s).
   - Không tạo cấu trúc dữ liệu lớn (RAM ≤ 128 MB).
   - Có nước dự phòng hợp lệ khi gặp exception.
4. Nhắc người dùng: nộp **đúng một file** `<MSSV>.py` (vd `26127000.py`) lên Moodle; không nộp zip, log hay báo cáo. Nếu file vẫn tên `MSSV.py`, nhắc đổi tên thành mã số sinh viên thật (hỏi mã số nếu chưa biết).
