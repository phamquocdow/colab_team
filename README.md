# Phát hiện tin giả (Fake / True News) – Khai phá dữ liệu văn bản

Tiền xử lý văn bản (tokenization, bỏ stop-word, lemmatization) → trích đặc trưng **TF-IDF** →
**phân cụm** bài báo theo chủ đề (K-Means) và **phân loại** Fake / True (**Naive Bayes**, **Passive-Aggressive**) →
phân tích **từ khóa có trọng số cao nhất** của mỗi lớp.

Dữ liệu: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) –
`Fake.csv` (23 481 bài) và `True.csv` (21 417 bài, toàn bộ từ Reuters), tin chính trị/thế giới 2016 – 2017.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `1_setup_moi_truong.ipynb` | Kiểm tra thư viện, tải tài nguyên NLTK, đưa dataset vào `data/raw/` |
| `2_eda.ipynb` | Khám phá dữ liệu: phân bố nhãn, độ dài, chuyên mục, dấu hiệu nguồn tin, từ phổ biến |
| `3_tien_xu_ly_van_ban.ipynb` | Tiền xử lý, khử bài rỗng/trùng lặp → `data/processed/news_clean.csv` |
| `4_hoc_co_giam_sat.ipynb` | Pipeline TF-IDF + NB / PA, chọn tham số bằng CV, đánh giá test, kiểm tra gần trùng, từ khóa, dự đoán bài mới |
| `5_hoc_khong_giam_sat.ipynb` | TF-IDF hướng chủ đề → LSA → K-Means; chọn K thủ công dựa trên các chỉ số, kiểm tra bài đại diện / ranh giới |
| `text_preprocessing.py` | Hàm tiền xử lý **dùng chung** cho huấn luyện và dự đoán bài mới |
| `topic_clustering.py` | Biểu diễn, khảo sát K không dùng nhãn, từ khóa có trọng số, bài đại diện và dự đoán chủ đề |
| `tests/test_topic_clustering.py` | Kiểm tra trọng số từ khóa, định danh bài và đầu vào không có đặc trưng |
| `data/raw/`, `data/upload/` | Dữ liệu gốc (`Fake.csv`, `True.csv`); `upload/` để đặt file zip tải tay nếu không tải được qua KaggleHub |
| `data/processed/` | Dữ liệu đã tiền xử lý (sinh bởi notebook 3) |
| `results/classification/`, `results/clustering/` | Bảng kết quả CSV (sinh bởi notebook 4, 5) |
| `models/` | Pipeline phân loại (notebook 4) và Pipeline TF-IDF → LSA → K-Means (notebook 5) |

> **Lưu ý khi clone từ git:** `data/` và `models/` không có trong repo (xem `.gitignore`) vì dung lượng lớn và đều sinh lại được.
> Chạy notebook 1 để tải dataset về `data/raw/`, sau đó chạy lần lượt notebook 3 → 5 để tạo `data/processed/` và `models/`.
> Notebook trong repo đã lưu sẵn output, và `results/` có đầy đủ bảng, biểu đồ kết quả, nên có thể xem kết quả mà không cần chạy lại.

## Cách chạy

**Yêu cầu:** Python 3.11, Windows/macOS/Linux; khuyến nghị 8 GB RAM. Thời gian chạy phụ thuộc máy;
POS tagging ở notebook 3, cross-validation ở notebook 4 và khảo sát nhiều seed ở notebook 5 là các bước tốn thời gian nhất.

**Cách 1 – Windows:** chạy `chay_jupyter.bat`. Lần đầu script tạo môi trường `.venv` và cài `requirements.txt`, sau đó mở JupyterLab.

**Cách 2 – thủ công:**

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Chạy **lần lượt notebook 1 → 5**, mỗi notebook dùng *Kernel → Restart Kernel and Run All Cells*
(notebook sau đọc file do notebook trước sinh ra). Hoặc chạy toàn bộ từ dòng lệnh:

```bash
for nb in 1_setup_moi_truong 2_eda 3_tien_xu_ly_van_ban 4_hoc_co_giam_sat 5_hoc_khong_giam_sat; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
```

## Dự đoán một bài báo mới

Bài mới phải đi qua **đúng hàm tiền xử lý đã dùng khi huấn luyện** (`preprocess_article`):

```python
import joblib
from text_preprocessing import preprocess_article

model = joblib.load("models/passive_aggressive.joblib")      # hoặc models/naive_bayes.joblib
x = preprocess_article("Tiêu đề bài báo", "Nội dung thân bài ...")
print(["Fake", "True"][model.predict([x])[0]])
```

## Tra chủ đề của bài mới

Sau khi chạy notebook 5:

```python
import joblib
import pandas as pd
from text_preprocessing import preprocess_article
from topic_clustering import predict_topics

model = joblib.load("models/topic_clustering.joblib")
x = preprocess_article("North Korea conducts missile test", "North Korea tested a new missile.")
cluster = int(predict_topics(model, [x])[0])
summary = pd.read_csv("results/clustering/cluster_summary.csv", index_col="cluster")
print("Không đủ đặc trưng" if cluster == -1 else summary.loc[cluster, "Chủ đề gợi ý"])
```

Tên gợi ý ghép từ các từ khóa; đối chiếu `representative_articles.csv` để đặt tên chủ đề chính thức.
Đầu vào không có từ trong từ vựng nhận `-1`. Mã cụm không mang ý nghĩa Fake/True.

Kiểm tra các trường hợp biên của phần phân cụm:

```bash
python -m unittest discover -s tests -v
```

## Các điểm phương pháp chính

- **Loại "dấu vân tay" nguồn tin** trước khi trích đặc trưng: dòng đề mục `(Reuters)`, dòng ghi công, câu tài trợ, link rút gọn của Reuters;
  chú thích ảnh `Featured image via…`, `Getty Images`, chân trang/quảng cáo của các trang tin giả… Nếu giữ lại, mô hình học "bài từ trang nào"
  thay vì nội dung (xem bảng kiểm tra cuối notebook 3).
- **Khử trùng theo thân bài** (không chỉ theo tiêu đề + thân bài) để một bài đăng lại với tiêu đề khác không rơi cả vào train lẫn test;
  bài chỉ có tiêu đề được giữ lại và đánh dấu ở cột `title_only`.
- **Không rò rỉ trong cross-validation:** TF-IDF nằm trong cùng `Pipeline` với classifier, `GridSearchCV` chạy trên văn bản – vocabulary/IDF
  của mỗi fold chỉ học từ phần train của fold đó. Tập test chỉ dùng một lần ở bước đánh giá cuối.
- **Kiểm tra bài gần trùng** giữa test và train (cosine TF-IDF) và báo cáo accuracy khi bỏ các bài này.
- Passive-Aggressive dùng `SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=C)` – thay thế chính thức của
  `PassiveAggressiveClassifier` (deprecated từ scikit-learn 1.8; ở bản 1.9 `GridSearchCV` không truyền được `C` vào lớp cũ).

## Kết quả chính

**Dữ liệu sau tiền xử lý:** 38 821 / 44 898 bài (20 935 True, 17 886 Fake). Đã loại 9 bài rỗng hoàn toàn và 6 068 bài trùng lặp
(5 857 bài trùng thân bài, 211 bài trùng tiêu đề + thân bài). Giữ lại 496 bài chỉ có tiêu đề (495 Fake, 1 True), đánh dấu ở cột `title_only`.

**Phân loại** (train 31 056 / test 7 765 bài, chia phân tầng 80/20):

| Mô hình | Tham số (5-fold CV) | F1-macro CV | Accuracy test | F1-macro test |
|---|---|---|---|---|
| Multinomial Naive Bayes | `alpha = 0.001` | 0.952 | 0.953 | 0.953 |
| Passive-Aggressive (PA-I) | `C = 0.03` | 0.990 | 0.990 | 0.990 |

- Chỉ 1.6 % bài test có bài train gần trùng (cosine ≥ 0.8). Bỏ các bài này, hoặc bỏ bài chỉ có tiêu đề, accuracy thay đổi không quá 0.0004.
- Từ khóa lớp True: `say`, `spokesman`, `news conference`, các ngày trong tuần, tên riêng quốc tế (`rohingya`, `puigdemont` …).
  Từ khóa lớp Fake: `video`, `watch`, `via`, `reportedly`, `gop`, nguồn dẫn lại (`via breitbart`, `wfb`, `nyp`), từ ngữ cảm xúc (`hilariously`).
  Phân tích chi tiết ở cuối notebook 4.

**Phân cụm theo chủ đề** (TF-IDF unigram + bigram, lọc từ văn phong → LSA 100 chiều → K-Means):

- Đã chạy đủ **K=2 đến 16**, ba seed (42, 17, 93), `n_init=10`, tổng cộng 45 lần thử. K-Means học trên **38.820 bài**; một bài không có đặc trưng nhận `-1`.
- **Chọn thủ công K=9**. Silhouette được tính trên cùng mẫu ngẫu nhiên cố định 3.000 bài; CH/DB dùng toàn bộ LSA đã chuẩn hóa. Nhãn Fake/True và `subject` chỉ được xem sau khi chốt K.
- Trung bình qua ba seed tại K=9: cosine silhouette **0,012154**, silhouette Euclidean LSA **0,056464**, CH **896,89**, DB **3,48726**, ARI ổn định **0,989994**. Std của cosine silhouette khoảng **0,00001379**, CH **0,00163**, DB **0,00056**.
- Mô hình cuối giữ đúng seed 42: cosine silhouette **0,012136**, silhouette Euclidean **0,056370**, CH **896,89**, DB **3,48653**, inertia **27.029,79**. `final_metrics.csv` ghi chỉ số mô hình này; `k_selection.csv` ghi tổng hợp các seed. ARI ổn định là chỉ số so các lần chạy với nhau, không phải độ chính xác 99%.

**Vì sao chọn K=9?**

| K | Cosine silhouette ↑ | CH ↑ | DB ↓ | ARI ổn định ↑ |
|---|---:|---:|---:|---:|
| 7 | 0,011770 | 980,41 | 3,65387 | 0,99014 |
| 8 | 0,011979 | 936,37 | 3,59472 | 0,98586 |
| **9** | **0,012154** | **896,89** | **3,48726** | **0,98999** |
| 10 | 0,012343 | 862,52 | 3,51421 | 0,98893 |
| 13 | 0,013039 | 775,20 | 3,48490 | 0,79221 |
| 16 | 0,013526 | 708,67 | 3,35219 | 0,66730 |

- So với K=7–8, K=9 có silhouette cao hơn và DB thấp hơn, vẫn giữ ARI gần 0,99; đổi lại CH giảm.
- So với K=10, CH tăng khoảng **3,98%**, DB giảm **0,77%**, ARI nhỉnh hơn và ít hơn một cụm. Chấp nhận cosine silhouette giảm khoảng **0,000189** (1,53% so với K=10), silhouette Euclidean giảm và inertia tăng. Không có kiểm định về ý nghĩa thống kê của các chênh lệch nhỏ; K=10 vẫn là một phương án có cơ sở.
- K=11–16 có ARI khoảng **0,579–0,792**, thấp hơn K=9. K=13 có DB gần K=9 nhưng dao động DB qua seed lớn hơn (std 0,05390 so với 0,00056). Vì vậy, ưu tiên độ ổn định thay vì chỉ tăng silhouette hoặc giảm DB.
- K=2 có CH cao nhất (1593,94) và K=4 có ARI cao nhất (0,99952), nhưng silhouette chỉ 0,007683/0,008040 và DB 4,83675/4,33881. Không chọn bằng cách tối đa một chỉ số hoặc đếm số chỉ số thắng; đây là lựa chọn khám phá dựa trên sự đánh đổi.
- K=16 có silhouette cao nhất và DB thấp nhất ở biên khảo sát. Chưa loại trừ K>16 phù hợp hơn; mở rộng dải là kiểm tra tiếp theo, không được khẳng định K=9 tối ưu toàn cục hay dữ liệu có đúng chín chủ đề.

**Chín cụm của mô hình cuối (seed 42):** tên gợi ý dựa trên từ khóa, tiêu đề và nội dung trích đoạn của bài đại diện; không phải nhãn chủ đề chuẩn.

| Cụm | Số bài | % dữ liệu | Tên chủ đề gợi ý | Nhận xét |
|---|---:|---:|---|---|
| 0 | 8.330 | 21,46 | Bình luận chính trị–xã hội Mỹ | Rộng; còn dấu hiệu văn phong và nội dung chồng lấn |
| 1 | 5.712 | 14,71 | Chính sách Mỹ: thuế, ngân sách, y tế | Nhiều chính sách trong một cụm; không chỉ riêng thuế |
| 2 | 2.803 | 7,22 | Chính trị châu Âu–EU–Brexit | Bài đại diện gồm Macedonia, Anh và Italy |
| 3 | 4.667 | 12,02 | Bầu cử Mỹ–Clinton và các ứng viên | Gồm cả tin ứng viên và bình luận; không chỉ email Clinton |
| 4 | 1.219 | 3,14 | Bán đảo Triều Tiên–hạt nhân–ngoại giao | Chủ đề rõ hơn theo từ khóa và ba bài đại diện |
| 5 | 7.287 | 18,77 | Tin quốc tế–Trung Đông, Iran/Syria, quân sự | Cụm lớn, phạm vi rộng; cần đọc thêm bài ở ranh giới |
| 6 | 2.849 | 7,34 | Cảnh sát–súng đạn–tội phạm tại Mỹ | Có cả tin tội phạm và bình luận xã hội |
| 7 | 2.820 | 7,26 | Tòa án–Tối cao Pháp viện–pháp luật Mỹ | Ba bài đại diện liên quan các vụ việc ở tòa án |
| 8 | 3.133 | 8,07 | Nga–FBI–điều tra can thiệp bầu cử Mỹ | Nhóm sự kiện/điều tra; chưa chứng minh mọi bài cùng chủ đề |

- Cụm lớn nhất: **8.330 bài (21,46%)**, nhỏ nhất: **1.219 bài (3,14%)**. Mã 0–8 chỉ thuộc mô hình hiện tại.
- Có **759/3.000 bài mẫu (25,30%) có cosine silhouette âm**. Cụm 0 và 5 có khoảng 57,44% và 51,23% bài mẫu điểm âm; còn chủ đề rộng và ảnh hưởng văn phong. Cụm Triều Tiên có silhouette trung bình mẫu khoảng 0,09940 và không có điểm âm trong 106 bài mẫu; không diễn giải thành chính xác 100%.
- Đối chiếu sau khi chốt mô hình: ARI với Fake/True **0,136176**, Cramér's V **0,646213**, purity **0,793637**, mốc lớp đông nhất **0,539284**. Các cụm có liên hệ với nhãn/nguồn; đây không phải độ chính xác chủ đề, không được dùng để chọn K.
- Mô hình nạp lại khớp 30 bài kiểm tra, đầu vào rỗng/không có đặc trưng nhận `-1`, bảng gán cụm đủ 38.821 bài. Chưa đánh giá trên nhãn chủ đề chuẩn hay nguồn dữ liệu mới.

**Câu trả lời ngắn khi bảo vệ:** “Em chọn K=9 sau khi khảo sát K=2–16 trên cùng dữ liệu. K=9 ổn định qua ba seed với ARI gần 0,99. So với K=8, silhouette cao hơn và DB thấp hơn; so với K=10, CH cao hơn, DB thấp hơn, ARI nhỉnh hơn và ít hơn một cụm, trong khi cosine silhouette chỉ thấp hơn khoảng 0,000189. Các K từ 11 trở lên có thể cải thiện một số chỉ số nhưng độ ổn định giảm rõ rệt. Vì vậy, K=9 là lựa chọn đánh đổi trong phạm vi khảo sát, chưa phải số chủ đề đúng duy nhất.”

Các đầu ra chính trong `results/clustering/`:
- `k_selection.csv`, `k_runs.csv`, `final_metrics.csv`: chỉ số tổng hợp theo K, kết quả từng seed và điểm mô hình đã chọn.
- CH (cao hơn tốt hơn) và DB (thấp hơn tốt hơn) được tính trên toàn bộ LSA đã chuẩn hóa; bảng tổng hợp có trung bình và độ lệch chuẩn qua ba seed. Silhouette dùng mẫu cố định 3.000 bài.
- `cluster_summary.csv`, `cluster_keywords.csv`: từ khóa, trọng số TF-IDF, điểm tương phản và tỷ lệ bài chứa từ.
- `representative_articles.csv`, `boundary_articles.csv`: ba bài gần tâm và hai bài gần ranh giới của mỗi cụm.
- `cluster_articles.csv`, `excluded_articles.csv`: gán cụm theo `raw_id`, khoảng cách và margin; margin **không phải xác suất**.
- `evaluation_sample.csv`, `run_config.json`, `topic_stop_words.txt`: mẫu đánh giá cố định và cấu hình để đối chiếu.
- `k_selection.png`, `cluster_overview.png`: biểu đồ chọn K, kích thước cụm, PCA minh họa và silhouette theo cụm.

## Hạn chế

- Mỗi lớp đến từ một nhóm nguồn cố định (True = Reuters, Fake = các trang bị PolitiFact/Wikipedia gắn cờ) và một số chuyên mục chỉ có ở một lớp
  (`worldnews`, `politicsNews` chỉ ở True). Từ khóa của mô hình cho thấy văn phong và chủ đề vẫn là tín hiệu rất mạnh, nên accuracy cao trên tập test
  **chưa chứng minh** mô hình nhận biết được tin đúng/sai ở nguồn tin khác – điều này **chưa được kiểm chứng**, cần đánh giá thêm trên dữ liệu
  ngoài tập hiện tại (ví dụ LIAR, FakeNewsNet, hoặc tin thật từ báo khác Reuters).
- Phân cụm dùng để khám phá dữ liệu: silhouette vẫn thấp, còn cụm rộng và ảnh hưởng của văn phong. Bộ lọc có thể loại cả từ mang nội dung;
  chưa có nhãn chủ đề chuẩn/người chấm độc lập để đo độ chính xác chủ đề. ARI ổn định chỉ kiểm tra độ nhạy theo seed, không thay cho
  kiểm tra trên mẫu dữ liệu khác. K được chọn trong lưới đã khảo sát, không phải số chủ đề thật duy nhất của dataset.
#   m y p j h i  
 