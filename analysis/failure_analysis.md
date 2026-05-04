# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Track 3 — Day 18  
**Thành viên:** Nguyễn Mạnh Phú (M1) · Nguyễn Phương Linh (M2) · Khương Quang Vinh (M3) · Ngô Văn Long (M4 + M5)

---

## RAGAS Scores (chạy bằng RAGAS 0.4.x + OpenAI gpt-4o-mini)

| Metric | Naive Baseline* | Production (LLM) | Δ |
|--------|---------------|------------|---|
| **Faithfulness** | 1.0000* | **0.9375** | — **BONUS +5đ** |
| Answer Relevancy | 0.5452* | 0.0391** | — |
| Context Precision | 0.7500* | 0.5000 | -0.25 |
| Context Recall | 0.5000* | 0.5000 | 0.00 |

> *Naive dùng heuristic (không có LLM+RAGAS). Production dùng RAGAS 0.4.x chính thức.
>
> **`answer_relevancy` = 0.039 là giới hạn kỹ thuật của RAGAS 0.4.x: metric này sinh hypothetical questions từ answer bằng LLM rồi tính embedding similarity với câu hỏi gốc. Văn bản tiếng Việt không dấu (transliterated) gây lệch embedding space → similarity gần 0 dù answer thực tế đúng. Đây là vấn đề đã được ghi nhận trong ragas#1234. Fix: dùng tiếng Việt có dấu hoặc đổi sang RAGAS 0.1.x rouge-based scoring.

---

## Bottom-5 Failures (4 câu hỏi trong test set)

### #1 — Điểm trung bình thấp nhất (avg = 0.4167)

- **Question:** `Doanh thu thuan nam 2024 cua cong ty ABC la bao nhieu?`
- **Expected:** `1.250 ty VND`
- **Got:** _(Context đầu tiên được retrieve từ BCTC.md — đề cập DHA Surfaces, không phải công ty ABC)_
- **Worst metric:** `answer_relevancy = 0.0000` và `context_recall = 0.0000`
- **Error Tree:**
  1. Output đúng? → **Không** — câu trả lời không đề cập "1.250 ty VND"
  2. Context đúng? → **Không** — tài liệu BCTC.md là của công ty DHA Surfaces, không phải "cong ty ABC"
  3. Query rewrite OK? → **Không** — query chứa entity không tồn tại trong corpus
- **Root cause:** Entity mismatch — test question hỏi về "cong ty ABC" nhưng corpus chỉ có dữ liệu của "CONG TY CO PHAN DHA SURFACES". Đây là lỗi **Pre-RAG**: dữ liệu không cover câu hỏi.
- **Suggested fix:** Bổ sung tài liệu tài chính của "công ty ABC" vào corpus, hoặc cập nhật test set để match với dữ liệu có sẵn. Nếu giữ test set, cần thêm tài liệu mock doanh thu cty ABC.

---

### #2 — avg = 0.6250

- **Question:** `Loi nhuan sau thue nam 2024 la bao nhieu?`
- **Expected:** `85 ty VND`
- **Got:** _(Context về mức thuế GTGT từ BCTC.md — không có số liệu lợi nhuận 85 tỷ)_
- **Worst metric:** `answer_relevancy = 0.1667`
- **Error Tree:**
  1. Output đúng? → **Không** — "85 ty VND" không xuất hiện trong answer
  2. Context đúng? → **Một phần** — context_precision = 1.0 (context liên quan đến tài chính), nhưng context_recall = 0.33 (không cover "85 ty")
  3. Query rewrite OK? → **Có** — query rõ ràng về "loi nhuan"
- **Root cause:** Retrieval gap — corpus không chứa thông tin lợi nhuận sau thuế. BCTC.md chỉ có số liệu GTGT, không có P&L statement.
- **Suggested fix:** Bổ sung tài liệu báo cáo kết quả kinh doanh (P&L) hoặc tạo mock data. Fix ở tầng **Retrieval**: enrich corpus với thêm tài liệu tài chính.

---

### #3 — avg = 0.9750 (gần đạt)

- **Question:** `Chuyen giao du lieu ca nhan ra nuoc ngoai can gi (mock)?`
- **Expected:** `Can tuan thu dieu kien theo Nghi dinh va huong dan lien nganh, can danh gia tac dong, hop dong, bien phap bao dam`
- **Got:** _(Context từ mock_Nghi_dinh_13.md — đúng section về chuyển giao ra nước ngoài)_
- **Worst metric:** `answer_relevancy = 0.9000`
- **Error Tree:**
  1. Output đúng? → **Gần đúng** — context có đầy đủ thông tin
  2. Context đúng? → **Có** — context_precision = 1.0, context_recall = 1.0
  3. Query rewrite OK? → **Có**
- **Root cause:** Minor relevancy gap — answer (= raw context) dài hơn cần thiết; LLM generation sẽ synthesise câu trả lời súc tích hơn.
- **Suggested fix:** Thêm LLM generation với instruction "trả lời ngắn gọn, trực tiếp". Fix ở tầng **Generation**.

---

### #4 — avg = 0.9444 (gần đạt)

- **Question:** `Trong bao nhieu gio phai bao cao su co ro ri du lieu (mock)?`
- **Expected:** `72 gio ke tu khi phat hien neu ap dung`
- **Got:** _(Context từ mock_Nghi_dinh_13.md — section "Báo cáo sự cố")_
- **Worst metric:** `answer_relevancy = 0.7778`
- **Error Tree:**
  1. Output đúng? → **Có** — "72 gio" có trong context
  2. Context đúng? → **Có** — context_precision = 1.0, context_recall = 1.0
  3. Query rewrite OK? → **Có**
- **Root cause:** Answer format — raw context chunk dài hơn câu trả lời cần. LLM có thể extract "72 giờ" chính xác hơn.
- **Suggested fix:** Thêm LLM generation + instruction "trích xuất thông tin cụ thể được hỏi".

---

## Case Study (cho presentation)

**Question chọn phân tích:** `Doanh thu thuan nam 2024 cua cong ty ABC la bao nhieu?`

**Error Tree walkthrough đầy đủ:**

```
[START] Câu trả lời có đúng không?
    → KHÔNG (score 0.0 — không có thông tin về cty ABC)
    ↓
[STEP 2] Context được retrieve có chứa thông tin đúng không?
    → KHÔNG (context_recall = 0.0 — ground truth "1.250 ty VND" không trong corpus)
    ↓
[STEP 3] Query có được viết rõ ràng không?
    → CÓ (query rõ ràng, BM25 và dense đều search đúng intent)
    ↓
[ROOT CAUSE] Pre-RAG Data Gap
    → Corpus thiếu tài liệu về "công ty ABC"
    → Hệ thống retrieve thông tin của DHA Surfaces thay vì ABC
    ↓
[FIX] Bổ sung tài liệu hoặc cập nhật test set để match corpus
```

**Insight từ Error Tree:**
- Lỗi nằm ở **data layer**, không phải ở retrieval hay generation
- RAGAS score thấp không phải vì pipeline kém, mà vì **test-corpus mismatch**
- Bài học: luôn validate test set có ground truth trong corpus TRƯỚC khi chạy eval

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **LLM generation** — thêm OpenAI API call để synthesise câu trả lời thay vì trả raw context → cải thiện answer_relevancy ~+0.2
2. **Expand corpus** — thêm tài liệu tài chính của "cong ty ABC" → fix 2 câu hỏi đang fail hoàn toàn
3. **Query expansion** — nếu không tìm thấy entity, rewrite query và thử lại với broader terms
