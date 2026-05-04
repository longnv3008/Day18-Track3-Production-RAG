# Individual Reflection — Lab 18

**Tên:** Ngô Văn Long (2A202600129)  
**Module phụ trách:** M4 — RAGAS Evaluation & Failure Analysis · M5 — Enrichment Pipeline

---

## 1. Đóng góp kỹ thuật

**Module đã implement:** `src/m4_eval.py`

**Các hàm/class chính đã viết:**

| Hàm | Mô tả |
|-----|-------|
| `_tokenize(text)` | Vietnamese-aware tokeniser, lọc stopwords tiếng Việt |
| `_heuristic_faithfulness()` | Token overlap giữa answer và contexts |
| `_heuristic_answer_relevancy()` | Keyword overlap giữa question và answer |
| `_heuristic_context_precision()` | Tỷ lệ contexts liên quan đến câu hỏi |
| `_heuristic_context_recall()` | Coverage của ground truth trong contexts |
| `_heuristic_evaluate()` | Heuristic RAGAS-style scoring (không cần LLM) |
| `evaluate_ragas()` | Dual-path: thử ragas library → fallback heuristic |
| `failure_analysis()` | Diagnostic Tree: tìm bottom-N, map → diagnosis + fix |

**Số tests pass:** 4/4 (`pytest tests/test_m4.py`)

**Ngoài M4, còn đóng góp:**
- Cập nhật `src/pipeline.py`: thêm LLM generation (OpenAI) + LatencyTracker
- Implement toàn bộ `src/m5_enrichment.py`: dual-path (OpenAI + extractive fallback) cho summarization, HyQA, contextual prepend, auto metadata
- Điền `analysis/failure_analysis.md` và `analysis/group_report.md`

---

## 2. Kiến thức học được

**Khái niệm mới nhất:**

**RAGAS 4 metrics** — bài giảng giới thiệu tổng quan, nhưng khi implement mới thấy rõ sự khác biệt:
- **Faithfulness** đo grounding: answer có dựa vào context không? (chống hallucination)
- **Answer Relevancy** đo utility: answer có trả lời câu hỏi không? (chống off-topic)
- **Context Precision** đo retrieval quality: bao nhiêu chunk retrieved là có ích?
- **Context Recall** đo coverage: ground truth có trong retrieved contexts không?

4 metrics này tạo ra một "diagnostic map" — khi một metric thấp là chỉ rõ vấn đề ở đâu trong pipeline.

**Điều bất ngờ nhất:**

Faithfulness = 1.0 khi answer = first context — nghe có vẻ tốt nhưng thực ra là **degenerate case**: hệ thống "honest" nhưng không hữu ích. Điều này minh họa tại sao cần đánh giá **tất cả 4 metrics cùng lúc**, không chỉ 1.

**Insight quan trọng nhất:** Test-corpus mismatch làm RAGAS scores thấp không phải vì pipeline kém. Bài học: validate data coverage trước khi đánh giá.

**Kết nối với bài giảng:**
- Slide về "RAG Failure Modes" → Error Tree trong failure_analysis()
- Slide về "Evaluation is not just accuracy" → 4 RAGAS metrics cover different failure modes
- Slide "Production RAG" → Dual-path implementation (official library + fallback)

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn lớn nhất:** RAGAS library yêu cầu OpenAI API key để chấm metrics. Không có key thì `evaluate_ragas()` fail hoàn toàn, dẫn đến pipeline không chạy được.

**Cách giải quyết:** Implement dual-path architecture:
1. **Primary path**: Thử ragas library với try/except đa tầng (handle cả API v0.1.x và v0.2.x)
2. **Fallback path**: Heuristic metrics thuần Python (sklearn tokenisation, TF-IDF overlap) — không cần API, không cần GPU, chạy instant

Design này đảm bảo pipeline luôn cho ra kết quả, ngay cả khi không có API key.

**Thách thức thứ 2:** ragas 0.2.x có API khác ragas 0.1.x (class-based metrics vs instance metrics). Giải quyết bằng cách thử import cả hai cách, lấy cái nào succeed.

**Thời gian debug:** ~45 phút cho phần RAGAS compatibility + 30 phút cho heuristic metrics calibration.

---

## 4. Nếu làm lại

**Sẽ làm khác:**
1. Bắt đầu với heuristic metrics ngay từ đầu (không phải refactor), vì đây là safety net quan trọng nhất
2. Validate test set coverage trước khi implement evaluation — phát hiện sớm hơn rằng 2/4 questions không có trong corpus
3. Thêm caching cho RAGAS calls để tránh re-run LLM khi test

**Module nào muốn thử tiếp:**
- **M2 (Hybrid Search)** — muốn hiểu sâu hơn về RRF scoring và việc tune `k` parameter
- **Fine-tuning BM25 tokeniser với underthesea** — ảnh hưởng lớn đến recall với tiếng Việt

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |

**Giải thích:**
- **Problem solving (5/5):** Dual-path architecture cho RAGAS là quyết định thiết kế không trivial, đảm bảo pipeline chạy được trong mọi điều kiện môi trường
- **Code quality (4/5):** Type hints đầy đủ, logic rõ ràng; trừ điểm vì chưa có unit test cho heuristic edge cases
- **Hiểu bài giảng (4/5):** Hiểu tốt 4 metrics và Error Tree; cần đọc thêm về RAGAS internals (LLM-based scoring mechanism)
- **Teamwork (4/5):** Ngoài M4, còn implement M5 và update pipeline để nhóm có thể chạy end-to-end
