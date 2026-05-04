# Individual Reflection — Lab 18

**Tên:** 2A202600467-Khương Quang Vinh 
**Module phụ trách:** [M3]

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **Module 3: Reranking**
- Các hàm/class chính đã viết: `CrossEncoderReranker` (implement logic tải model và rerank), `benchmark_reranker` (đo lường hiệu năng).
- Số tests pass: **5/5** (test_rerank_returns, test_rerank_type, test_rerank_sorted, test_rerank_relevant_first, test_benchmark_stats).

## 2. Kiến thức học được

- Khái niệm mới nhất: Sự khác biệt giữa **Bi-encoder** (nhanh nhưng ít chính xác hơn) và **Cross-encoder** (chậm hơn nhưng cực kỳ chính xác vì so khớp đồng thời query và document).
- Điều bất ngờ nhất: Điểm số của Cross-encoder phân tách cực kỳ rõ ràng giữa câu trả lời đúng (0.9914) và các câu trả lời sai (< 0.03).
- Kết nối với bài giảng (slide nào): Slide về Reranking - Highest ROI Optimization

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: 
    - Việc xử lý các lỗi xung đột phiên bản thư viện khi chạy RAGAS và cài đặt              `sentence-transformers` trong môi trường Conda.
    - xử lý file pdf scan ocr
- Cách giải quyết: fix lỗi ocr, Kiểm tra lại các dependencies trong `requirements.txt`, cập nhật đúng package và sử dụng explicit imports để tránh lỗi `AttributeError`.
- Thời gian debug: ~45 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thử nghiệm thêm các mô hình Reranker siêu nhẹ (như Flashrank hoặc mô hình nhỏ của Cohere) để tối ưu hóa thời gian phản hồi cho ứng dụng thực tế.
- Module nào muốn thử tiếp: Module 4 (Evaluation) để định lượng chính xác sự cải thiện của Reranker đối với các chỉ số như `Context Precision`.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |
