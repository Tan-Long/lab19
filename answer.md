# Phần 1: Nghiên cứu và chuẩn bị — Gợi ý trả lời (GraphRAG)

## Mục tiêu bài học (tóm tắt)

- Trích xuất **thực thể** và **quan hệ** từ văn bản thô.
- Làm quen các thư viện/công cụ quản lý đồ thị: **NetworkX**, **Neo4j**, framework **NodeRAG**.
- Xây dựng pipeline **GraphRAG** hoàn chỉnh: từ **indexing** đến **multi-hop querying**.
- **Đánh giá** sự khác biệt về độ chính xác giữa **Flat RAG** và **GraphRAG**.

---

## 2.1. Quy trình xử lý dữ liệu đồ thị

### 1. Entity extraction: LLM phân biệt **entity (node)** và **attribute** thế nào?

- **Entity (node)**: là “đối tượng” có danh tính tương đối ổn định trong ngữ cảnh bài toán — thường là người, tổ chức, địa danh, sản phẩm, khái niệm có thể đứng độc lập và **nối quan hệ** với entity khác. Trong đồ thị, entity thường được biểu diễn bằng **một nút** (có thể kèm loại/type).
- **Attribute**: là **tính chất gắn với một entity** (tuổi, ngày, màu sắc, số lượng, trạng thái…) — thường không cần là nút riêng mà là **thuộc tính** trên nút hoặc trên cạnh, tùy thiết kế schema.

LLM không “tự biết” ranh giới tuyệt đối; thực tế phân biệt dựa trên:

- **Prompt và schema rõ ràng** (ví dụ: chỉ cho phép các loại entity định nghĩa trước; quy tắc “chỉ tạo node khi có chỉ danh rõ và có thể liên kết”).
- **Định dạng đầu ra có cấu trúc** (JSON) để tách `entities` và `properties` / `attributes`.
- **Hậu xử lý**: chuẩn hóa tên, gộp trùng (**deduplication** / entity resolution).

### 2. Graph construction: Tại sao **deduplication** quan trọng trong đồ thị?

- **Tránh nhiều nút cho cùng một thực thể** (khác chính tả, alias, viết hoa/thường) — làm đồ thị phình to, nhiễu, khó truy vấn.
- **Tập trung quan hệ đúng chỗ**: mọi cạnh (làm việc tại, sở hữu, thuộc về…) phải trỏ về **một** đại diện chuẩn của entity.
- **Cải thiện multi-hop và BFS**: duyệt đồ thị chỉ hữu ích khi các bước đi qua **đúng nút**; dedup + entity resolution là nền tảng cho retrieval có cấu trúc.

### 3. Query answering: **BFS** trên đồ thị khác **vector search** thường ở điểm nào?

- **Vector search**: tìm các **chunk/đoạn văn** có embedding “gần” câu hỏi — mạnh về **tương đồng ngữ nghĩa mềm**, nhưng **không bắt buộc** theo cấu trúc quan hệ (ai với ai, thuộc tổ chức nào, chuỗi sự kiện theo loại cạnh…).
- **BFS (breadth-first search) trên đồ thị**: mở rộng từ (các) nút gốc theo **từng tầng hop**, đi theo **cạnh có kiểu quan hệ** — phù hợp câu hỏi cần **suy luận qua nhiều bước** khi tri thức đã được mô hình hóa thành graph.

Trong GraphRAG thực tế, hai hướng thường **kết hợp**: vector để chọn **điểm vào** (seed chunk/entity), duyệt đồ thị (BFS hoặc chiến lược có giới hạn) để **mở rộng theo cấu trúc**.

---

## 2.2. Tìm hiểu công cụ

| Công cụ | Mô tả ngắn |
|--------|------------|
| **NetworkX** | Thư viện Python cho đồ thị và mạng phức tạp; phù hợp **prototype** và thử nghiệm nhanh không cần CSDL đồ thị riêng. |
| **Neo4j** | Cơ sở dữ liệu đồ thị phổ biến trong ngành; truy vấn bằng **Cypher**; phù hợp dữ liệu lớn và triển khai production. |
| **NodeRAG** | Framework mã nguồn mở xây trên NetworkX; mục tiêu **đơn giản hóa** việc tích hợp GraphRAG vào ứng dụng Python. |

---

## Đồ án: Knowledge Graph + GraphRAG (kịch bản thử nghiệm)

### Corpus

- **100 bài Wikipedia** về các **công ty AI**.
- Trích xuất **thực thể** từ các bài viết và lưu vào **đồ thị Neo4j** (knowledge graph).

### So sánh hai pipeline (Flat RAG vs GraphRAG)

| Truy vấn | Đặc điểm | Kết quả ghi nhận |
|----------|-----------|------------------|
| **“What is OpenAI?”** | Câu hỏi đơn giản, gần như tra cứu trực tiếp trong văn bản | **Cả hai pipeline đều đúng** — Flat RAG và GraphRAG đều trả lời chính xác. |
| **“AI companies co-founded by former Google employees”** | Câu hỏi **multi-hop** (cần nối nhiều mối quan hệ: người → từng làm Google → đồng sáng lập → công ty AI) | **Flat RAG dễ hallucinate**; **GraphRAG trả lời đúng** nhờ retrieval theo cấu trúc đồ thị. |

### Trực quan hóa (Neo4j Browser)

- Dùng **Neo4j Browser** để:
  - **Hiển thị subgraph** liên quan đến câu trả lời.
  - **Highlight** các **nút (nodes)** tương ứng phần trả lời (answer nodes).

---

## Mục tiêu & sản phẩm bàn giao (deliverable)

### Mục tiêu

- Xây **Knowledge Graph** và **GraphRAG agent**.
- **Độ chính xác multi-hop** của GraphRAG **vượt Flat RAG ít nhất +20%** (mục tiêu định lượng trong đề bài).

### Deliverable

- **Visualization** của knowledge graph đã xây.
- **Báo cáo benchmark GraphRAG**, gồm tối thiểu:
  - **Multi-hop accuracy**
  - **Latency** (độ trễ)
  - **Cost** (chi phí gọi mô hình / hạ tầng)

### Thời gian

- Khung thời gian dự kiến trong tài liệu: khoảng **2 tuần** (phần cuối dòng trong slide có thể bị cắt).

---

## Quy trình triển khai GraphRAG (4 bước)

### 1. Entity extraction

- Áp dụng **NER dựa trên LLM** trên corpus theo miền (domain).
- **Đầu ra:** các bộ ba dạng **(subject, predicate, object)** — triples phục vụ dựng đồ thị.

### 2. Build graph

- Nạp triples vào **NetworkX** (prototype) hoặc **Neo4j** (lưu trữ, truy vấn Cypher).
- Bổ sung **embedding cho các nút** (node embeddings) để hỗ trợ tìm kiếm / gợi ý **seed** theo vector khi cần.

### 3. GraphRAG retrieval

Luồng xử lý gợi ý:

`query` → **seed nodes** → **BFS traversal** (duyệt đồ thị) → **subgraph-to-text** (chuyển subgraph thành ngữ cảnh văn bản) → **LLM generate** (sinh câu trả lời cuối).

### 4. Benchmark

- So sánh **GraphRAG** và **Flat RAG** trên **20 câu hỏi multi-hop**.
- Đo **accuracy**, **latency**, **cost** cho từng phương án và tổng hợp trong báo cáo.

---

## Bài lab: từ indexing đến đánh giá (theo slide thực hành)

### Bước 1: Trích xuất thực thể và quan hệ (Indexing)

- Dùng **LLM** đọc bộ dữ liệu **“Tech Company Corpus”** và chuyển thành các **bộ ba (triples)**.

**Ví dụ**

- **Input:**  
  `OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015.`

- **Output (triples):**
  - `(OpenAI, FOUNDED_BY, Sam Altman)`
  - `(OpenAI, FOUNDED_BY, Elon Musk)`
  - `(OpenAI, FOUNDED_IN, 2015)`

---

### Bước 2: Xây dựng đồ thị (Construction)

Đưa triples vào **một trong ba** hướng sau:

| Lựa chọn | Công cụ | Khi nào dùng |
|----------|---------|----------------|
| **A** | **NetworkX** | Phù hợp chạy **offline trong Notebook**. |
| **B** | **Neo4j** | **Khuyên dùng** nếu muốn **trực quan hóa** các mối liên kết rõ ràng (Neo4j Browser / Bloom). |
| **C** | **NodeRAG** | Khi cần giải pháp **trọn gói (all-in-one)** với logic tìm kiếm đã được tối ưu sẵn. |

---

### Bước 3: Thực thi truy vấn (Querying)

Viết hàm xử lý truy vấn theo logic:

1. **Nhận câu hỏi** từ người dùng.
2. **Trích xuất thực thể chính** trong câu hỏi (ví dụ: `"Google"`).
3. **Tìm node** tương ứng trong đồ thị và **duyệt (traverse)** các node lân cận trong phạm vi **2-hop**.
4. **Gộp** thông tin thu được thành một đoạn văn (**textualization**) và **gửi cho LLM** để sinh câu trả lời.

---

### Bước 4: So sánh và đánh giá (Evaluation)

Chạy thử **5 câu hỏi phức tạp** trên **cả hai** hệ thống:

1. **Flat RAG:** chỉ dùng **ChromaDB** hoặc **Faiss** (vector store + chunk).
2. **GraphRAG:** dùng **đồ thị** vừa xây dựng.

**Yêu cầu:** ghi lại các trường hợp **Flat RAG bị ảo giác (hallucination)** nhưng **GraphRAG trả lời đúng** — kèm câu hỏi, câu trả lời hai phía và nhận xét ngắn.

---

## 5. Đề xuất công cụ (Recommendations)

| Mục tiêu | Tool gợi ý | Lý do |
|----------|------------|--------|
| **Dễ bắt đầu** | **NodeRAG** | Tích hợp sẵn logic GraphRAG, không cần cấu hình database phức tạp. |
| **Trực quan hóa tốt nhất** | **Neo4j** | Giao diện đồ họa giúp “thấy” được tri thức đang được kết nối như thế nào (Browser/Bloom). |
| **Nghiên cứu thuật toán** | **NetworkX** | Cho phép can thiệp sâu vào các thuật toán toán học của đồ thị. |

---

## 6. Deliverables (nộp bài)

Sinh viên nộp báo cáo **bao gồm**:

1. **Mã nguồn** (file `.py` hoặc `.ipynb`).
2. **Ảnh chụp màn hình** đồ thị tri thức đã xây dựng (**Neo4j** hoặc **Matplotlib**).
3. **Bảng so sánh** kết quả **20 câu hỏi benchmark** giữa **Flat RAG** và **GraphRAG**.
4. **Phân tích ngắn gọn** về **chi phí** khi xây dựng đồ thị: **token usage**, **thời gian (time)**.

---

*Tài liệu gộp phần nghiên cứu (2.1–2.2), đồ án/kịch bản, quy trình & lab, đề xuất công cụ (§5), checklist nộp bài deliverables (§6).*
