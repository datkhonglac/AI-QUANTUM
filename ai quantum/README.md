
# AI–Quantum Portfolio Intelligence Platform

Proof of Concept cho cuộc thi AI–Quantum Challenge 2026.

## Mục tiêu

- Dự báo lợi nhuận kỳ vọng bằng học máy.
- Mã hóa bài toán lựa chọn tài sản dưới dạng QUBO.
- Mô phỏng QAOA p=1 để chọn đúng K tài sản.
- Tối ưu tỷ trọng bằng phương pháp cổ điển trong tập tài sản đã chọn.
- So sánh với equal-weight và baseline cổ điển.
- Hiển thị lợi nhuận, rủi ro, drawdown và giải thích lựa chọn.

> Đây là sản phẩm nghiên cứu, không phải khuyến nghị đầu tư.

## Kiến trúc

`Data → AI Forecast → QUBO → QAOA Selection → Classical Weight Optimization → Backtest → Dashboard`

## Chạy trên máy tính

Yêu cầu Python 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

## Định dạng dữ liệu

CSV dạng wide:

```csv
Date,FPT,HPG,MBB,MWG,SSI,VNM,VCB,GAS
2021-01-04,....
2021-01-05,....
```

File `data/demo_prices.csv` là dữ liệu mô phỏng để bảo đảm demo chạy ổn định. Khi nộp chính thức, nhóm nên thay hoặc bổ sung dữ liệu lịch sử hợp pháp và ghi rõ nguồn.

## Kiểm thử

```bash
pytest -q
```

## Đưa mã lên GitHub

1. Tạo repository public, ví dụ `ai-quantum-portfolio-platform`.
2. Giải nén toàn bộ thư mục dự án vào repository.
3. Commit và push lên nhánh `main`.
4. Dùng URL repository làm **Link GitHub / mã nguồn**.

## Deploy Streamlit

1. Đăng nhập Streamlit bằng GitHub.
2. Tạo ứng dụng từ repository.
3. Chọn nhánh `main`.
4. Chọn entry file `app.py`.
5. Deploy và kiểm tra ở chế độ ẩn danh.
6. Dùng URL ứng dụng làm **Link demo sản phẩm**.

Giao diện nền tảng có thể thay đổi theo thời điểm; các trường cốt lõi vẫn là repository, branch và entry file.

## Phạm vi khoa học

- QAOA chạy bằng statevector simulator ở quy mô nhỏ.
- Không tuyên bố quantum speedup.
- QAOA chọn tập tài sản; tối ưu cổ điển xác định tỷ trọng cuối cùng.
- Baseline cổ điển được giữ để so sánh công bằng.
