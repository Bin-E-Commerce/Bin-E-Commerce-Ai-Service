# Search Cross-Encoder

Search lấy candidate từ hai nguồn: PostgreSQL tìm theo tên, brand, danh mục,
variant, mã và fuzzy text; Qdrant tìm các sản phẩm gần nghĩa từ embedding truy
vấn. Product Service xác minh lại mọi candidate bằng các bộ lọc catalog trong
PostgreSQL, gộp tối đa 100 sản phẩm rồi Cross-Encoder chấm lại thứ tự trước khi
phân trang. PostgreSQL cung cấp phần đuôi để các trang sau vẫn có thể tiếp tục.
Mã SKU/GTIN và tên khớp chính xác giữ ưu tiên xác định. Candidate được gán điểm
relevance từ 0 đến 3; điểm ONNX là relevance score thô, không phải xác suất mua
hàng.

## Dữ liệu cần gán nhãn

Mỗi dòng JSONL là một truy vấn và tập candidate lấy từ kết quả thật của bộ tìm
kiếm. Hãy đánh giá toàn bộ candidate trong tập đó, không chỉ sản phẩm được click.
Không tạo nhãn dựa trên output của model đang được đánh giá.

```json
{"queryId":"q-0001","query":"điện thoại pin lâu","candidates":[{"itemId":"product-1","text":"Điện thoại ...","relevance":3,"retrievalRank":1},{"itemId":"product-2","text":"Pin dự phòng ...","relevance":1,"retrievalRank":2},{"itemId":"product-3","text":"Áo khoác ...","relevance":0,"retrievalRank":3}]}
```

Label guide:

- `0`: không liên quan.
- `1`: có liên quan nhẹ, chẳng hạn phụ kiện bổ trợ nhưng không đáp ứng ý định chính.
- `2`: phù hợp với ý định và có thể thay thế.
- `3`: đáp ứng trực tiếp ý định tìm kiếm.

`retrievalRank` là thứ tự pool ứng viên trước Cross-Encoder: kết quả PostgreSQL
theo relevance trước, các sản phẩm semantic mới được nối theo thứ tự Qdrant.
Đây là baseline công bằng trên cùng candidate set; nó không phải baseline
PostgreSQL-only. `text` nên được tạo nhất quán từ tên, brand, category, SKU và
mô tả public.
Không đưa thông tin cá nhân, query từ một người dùng cụ thể, token, email hay
địa chỉ vào file. Chia query thành ba file độc lập: `train.jsonl`,
`validation.jsonl`, `test.jsonl`; query text phải không trùng giữa các split.
Ưu tiên relevance judgments do người đánh giá gán. Click thuần có thiên lệch vị
trí hiển thị; nếu dùng click log thì cần lưu impression/position và xử lý bias
trước khi coi click là nhãn relevance.

## Huấn luyện và đo

Lệnh dưới đây dùng model multilingual mMARCO làm điểm khởi đầu, fine-tune bằng
pairwise ranking loss, rồi so sánh với thứ tự pool trước reranking trên đúng
candidate/qrels của test. Script đánh giá từng epoch trên validation, chọn
checkpoint có NDCG@10 cao nhất (MRR và Recall@10 dùng để phá hòa), đồng thời giữ
model gốc nếu fine-tune không cải thiện validation. Test chỉ được mở sau bước
chọn model:

```powershell
cd services/ai-service
python -m pip install -e ".[search-training]"
python -m app.modules.search_ranking.infrastructure.training.train_search_reranker `
  --train data/search/train.jsonl `
  --validation data/search/validation.jsonl `
  --test data/search/test.jsonl `
  --output artifacts/search-reranker
```

Report ghi `NDCG@10`, `Recall@10`, `MRR`, số query và chênh lệch với baseline
trước reranking vào `evaluation-report.json`. Test chỉ được đo sau khi hoàn tất huấn
luyện; không dùng test để chọn epoch hoặc tham số. Kết quả chỉ có ý nghĩa đối
với qrels đã gán nhãn và candidate generation tương ứng. Lệnh cũng lượng tử hóa
artifact sang dynamic INT8 và đo lại chính artifact đó để metric khớp với model
được deploy.

Artifact gồm `search-reranker.onnx`, sidecar `.meta.json` và
`hf-model/tokenizer.json`. Cấu hình AI Service:

```env
SEARCH_RERANKER_MODEL_PATH=/app/artifacts/search-reranker.onnx
SEARCH_RERANKER_TOKENIZER_PATH=/app/artifacts/search-reranker/hf-model/tokenizer.json
SEARCH_RERANKER_MODEL_VERSION=search-reranker-v1
```

Khi chưa mount artifact, service báo model chưa sẵn sàng và Product Service giữ
thứ tự PostgreSQL, không gọi query embedding. Khi model sẵn sàng nhưng OpenAI,
Qdrant hoặc embedding cache gặp lỗi, Product Service vẫn rerank candidate SQL
nếu có thể; lỗi Cross-Encoder sẽ fallback hoàn toàn về PostgreSQL. Chỉ 100
candidate đầu tiên được rerank; các trang sau lấy phần đuôi lexical sau khi loại
candidate pool. Do Qdrant chỉ trả một tập giới hạn, tổng số kết quả semantic là
số candidate đủ điều kiện đã thu được, không phải tổng số mọi sản phẩm gần nghĩa
trong catalog.

Chưa có model artifact được huấn luyện từ qrels tiếng Việt trong repository và
chưa có kết quả đánh giá thực nghiệm. Không dùng dữ liệu tổng hợp làm bằng chứng
hiệu quả với người dùng thật. Cần đánh giá riêng candidate recall trước khi kết
luận về chất lượng reranker, vì model không thể xếp hạng sản phẩm chưa được
retrieval lấy vào candidate pool.

## Cơ sở mô hình

- mMARCO cung cấp dữ liệu passage ranking đa ngôn ngữ để khởi tạo trọng số; nó
  không thay thế relevance judgments riêng cho catalog tiếng Việt:
  [Bonifacio et al., mMARCO](https://arxiv.org/abs/2108.13897).
- Cross-Encoder chấm query và document cùng lúc; thường rerank một tập ứng viên
  nhỏ sau retrieval:
  [Sentence Transformers: Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
- Đánh giá reranking trên tập ứng viên của bước retrieval và chọn model theo
  validation là quy trình được thư viện hỗ trợ:
  [CrossEncoderRerankingEvaluator](https://www.sbert.net/docs/package_reference/cross_encoder/evaluation.html),
  [Cross-Encoder training overview](https://www.sbert.net/docs/cross_encoder/training_overview.html).
- Pairwise ranking tối ưu thứ tự tương đối, kế thừa ý tưởng từ RankNet:
  [Burges et al., From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf).
