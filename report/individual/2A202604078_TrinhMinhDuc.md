# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

MSSV: 2A202604078
Họ và tên:Trịnh Minh Đức 
SEED: 3413

## 1. Bằng chứng

| Tệp | Tình trạng và nội dung đối chiếu |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | Có code, biểu đồ và output; ô báo cáo cuối ghi `SEED=3413`, thử thách #6, **15/15 kiểm tra đạt**, đủ 10 dự đoán. Một số output chưa đồng nhất với code hiện tại, được nêu ở mục 4. |
| [`lab1_report.json`](lab1_report.json) | Có `seed`, `checks`, `predictions`, `experiments`, `challenge`, `results`; lưu kết quả của 8 phương án. |
| [`hand_calc.jpg`](hand_calc.jpg) | Có ảnh viết tay với các bước của 5 phép tính. Phép baseline cần sửa số liệu đầu vào như giải thích bên dưới. |

### Dữ liệu và cách đánh giá

Dữ liệu gốc có **5.572 SMS**, gồm **4.825 ham** và **747 spam**; phát hiện **403 dòng trùng văn bản**. Sau khi bỏ trùng còn **5.169 tin**, gồm **4.516 ham** và **653 spam**. Với SEED 3413, notebook chia **3.619 train** (462 spam), **775 validation** và **775 test** (106 spam). SUB lấy từ test, gồm **20 spam + 20 ham**, dùng chung để so sánh với LLM.

Tin trùng nằm ở cả train và test làm điểm đánh giá lạc quan vì mô hình đã gặp chính nội dung đó khi học. Bỏ trùng tuyệt đối trước khi chia giúp giảm rò rỉ, nhưng các mẫu spam chỉ khác mã thưởng hoặc chi tiết nhỏ vẫn có thể xuất hiện ở cả hai tập.



## 2. Kết quả chính

### Bảng so sánh cuối Phần 5

Số liệu lấy từ `results` trong JSON, giữ thêm chữ số ở cột chi phí để tránh làm tròn mất khác biệt. Lớp dương là **spam**.

| Phương án | Precision | Recall | F1 test 775 | F1 SUB 40 | Train/chuẩn bị (s) | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.9886 | 0.8208 | 0.8969 | 0.9189 | 3.089 | 0.0488 | 0 |
| 2. BoW + MLP | 0.9897 | 0.9057 | 0.9458 | 0.9474 | 3.650 | 0.2652 | 0 |
| 3a. MiniLM + kNN | 0.9279 | 0.9717 | 0.9493 | 1.0000 | 50.918 | 64.0050 | 0 |
| 3b. MiniLM + LogReg | 0.9358 | 0.9623 | 0.9488 | 0.9744 | 51.141 | 34.8194 | 0 |
| 3c. MiniLM + MLP head | 0.9806 | 0.9528 | **0.9665** | 0.9744 | 51.656 | 37.2382 | 0 |
| 4a. LLM nhỏ zero-shot | 0 | 0 | Chưa đo | 0 | 0 | 1635.1522 | 0.006497 |
| 4b. LLM lớn zero-shot | 0 | 0 | Chưa đo | 0 | 0 | 1431.7008 | 0.064967 |
| 4c. LLM nhỏ few-shot | 0 | 0 | Chưa đo | 0 | 0 | 1409.6235 | 0.016363 |

**Phạm vi đo:** LLM chỉ chạy SUB 40; trường `f1` của các hàng LLM trong JSON cũng là kết quả SUB, không phải test 775. Precision/recall của hàng 1–3 tính trên test 775; của hàng 4 tính trên SUB. Độ trễ tính từ văn bản thô đến nhãn, gồm vector hóa/encode; thời gian chuẩn bị của MiniLM gồm encode tập train và huấn luyện head nếu có. `USD/1.000 tin = 0` của mô hình cục bộ nghĩa là không có phí API, vẫn có chi phí phần cứng và vận hành.

### Nhận xét từ số liệu

BoW + LogReg đạt accuracy **0.9742**, nhưng recall chỉ **0.8208**: trên test có **TP=87, FP=1, FN=19, TN=668**. Precision cao cho thấy ít chặn nhầm ham, còn 19/106 spam bị bỏ lọt cho thấy accuracy chưa đủ để đánh giá bộ lọc. Baseline luôn đoán ham cũng đạt khoảng 87% accuracy trên dữ liệu đã làm sạch nhưng không bắt được spam nào.

Thêm MLP trên BoW làm F1 tăng **0.0489**, từ 0.8969 lên 0.9458; trên embedding, thêm MLP head làm F1 tăng **0.0177**, từ 0.9488 lên 0.9665. Encoder đã cung cấp biểu diễn ngữ nghĩa, nên lợi ích thêm lớp ẩn nhỏ hơn trong lượt chạy này. Đây là kết quả quan sát; cần nhiều lần chạy hoặc đánh giá thêm để kết luận chắc chắn về mức cải thiện.

MLP BoW có **128.129 tham số**, so với **2.001** của LogReg. Output huấn luyện ghi **validation loss thấp nhất ở epoch 15**; đây là mốc phù hợp để cân nhắc early stopping, thay cho nhận xét chung “epoch 8–12” trong phần trả lời cũ. Mạng bỏ ReLU cũng đạt F1 **0.946** khi làm tròn: hai lớp affine liên tiếp gộp được thành một lớp affine, nhưng cách tối ưu khác nên kết quả không bắt buộc trùng LogReg numpy.

Ở thí nghiệm learning rate, loss cuối sau 300 epoch lần lượt là **0.627** (`lr=0.001`), **0.086** (`lr=0.5`) và **0.005** (`lr=50`). Đồ thị `lr=50` có một đỉnh loss lớn ở đầu rồi mới giảm nhanh; vì vậy không thể mô tả toàn bộ quá trình là giảm mượt ngay từ đầu hoặc suy rộng rằng learning rate lớn luôn tốt.

**Giới hạn quan trọng của LLM:** cả 4a, 4b và 4c đều ghi **`invalid=40/40`**. Hàm `llm_classify_batch` thay nhãn không parse được bằng `ham`, nên toàn bộ dự đoán thành ham: accuracy **0.5**, recall/F1 spam **0** trên SUB cân bằng. Đây là lỗi của luồng gọi/đọc kết quả trong lượt chạy, chưa phản ánh năng lực phân loại thực của các LLM. Notebook chưa lưu đủ phản hồi thô và `finish_reason` để khẳng định nguyên nhân; giới hạn 3 token là điểm cần kiểm tra, trong khi code hiện tại đã có cấu hình `reasoning_effort="none"`.

### Ba lỗi phương pháp và logic

| Bài bắt lỗi | Lỗi | Hậu quả và cách xử lý |
| --- | --- | --- |
| 1 | Đánh giá trên chính dữ liệu train; xây vocab từ toàn bộ `raw` chưa bỏ trùng thay vì chỉ train | Điểm không đo khả năng tổng quát hóa; tin trùng và thống kê từ vựng có thể làm đánh giá lạc quan. Bỏ trùng, chia train/val/test, rồi xây vocab bằng train và chỉ đánh giá cuối trên test. |
| 2 | `M @ q` chưa chia norm; `argsort(scores)[:k]` lấy điểm nhỏ nhất | Tích vô hướng bị ảnh hưởng bởi độ dài vector, còn sắp xếp sai chọn các tin ít giống nhất. Cần cosine có xử lý vector 0 và lấy k điểm cao nhất. |
| 3 | Kiểm tra chuỗi con `"spam" in response.lower()` | `"not spam"` vẫn chứa `"spam"`, nên ham bị chặn nhầm. Cần kiểm tra nhãn hợp lệ rõ ràng và ghi nhận đầu ra không hợp lệ thay vì mặc nhiên coi là ham. |

## 3. Dự đoán và thực tế

### Dấu thời gian của 10 dự đoán

Các mốc dưới đây chép từ JSON; cột thí nghiệm dùng lần chạy đầu của mỗi khóa. Cả 10 dự đoán đều được ghi trước thí nghiệm tương ứng theo log.

| Khóa | Ghi dự đoán | Chạy thí nghiệm đầu tiên |
| --- | --- | --- |
| `ml_f1` | 09:21:02 | 09:21:15 |
| `ml_lr` | 09:26:17 | 09:26:27 |
| `dl_curve` | 09:38:09 | 09:38:16 |
| `dl_relu` | 09:41:38 | 09:41:44 |
| `emb_search` | 09:49:30 | 09:49:36 |
| `emb_tricky` | 09:52:45 | 09:52:53 |
| `llm_models` | 10:10:51 | 10:10:55 |
| `llm_fewshot` | 10:12:30 | 10:12:34 |
| `tradeoff` | 10:18:40 | 10:18:51 |
| `hard_eval` | 10:21:59 | 10:22:19 |

### Ba dự đoán lệch đáng chú ý

**1. `emb_tricky`: embedding sẽ đúng nhiều hơn BoW.** Thực tế cả hai cùng đúng **5/6**, cùng bỏ lọt câu “Last chance! Your account rewards expire tonight, confirm details to keep them”. BoW thiếu các tín hiệu từ khóa đủ mạnh; embedding có thể đặt câu này gần thông báo tài khoản hợp lệ. Giải thích vị trí embedding là giả thuyết theo nội dung câu, chưa có phân tích láng giềng để xác nhận. Có biểu diễn ngữ nghĩa không bảo đảm phân biệt được ý đồ lừa đảo ở mọi câu.

**2. `llm_models`: model lớn sẽ hơn khoảng 0.05–0.10 F1.** Thực tế cả hai F1 bằng **0** do 40/40 đầu ra không parse được, nên không kiểm chứng được dự đoán về chất lượng. Chi phí ước tính model lớn đúng là **gấp 10 lần** model nhỏ theo cấu hình lab, nhưng cần sửa và xác minh luồng đầu ra trước khi so năng lực. Few-shot cũng không cải thiện F1 trong tình trạng này; chi phí tăng khoảng **2.52 lần** so với zero-shot nhỏ, thấp hơn dự đoán 3–4 lần.

**3. `tradeoff`: kNN sẽ bị 3b/3c lấn át.** Biểu đồ thực tế dùng **F1 SUB 40**, và output ghi đường Pareto gồm **1, 2, 3b, 3a**. kNN đạt F1 SUB **1.0000**, cao hơn 3b/3c (**0.9744**), nên vẫn nằm trên đường Pareto dù p95 **64.005 ms** vượt SLA. 3c bị 3b lấn át trên SUB vì cùng F1 nhưng chậm hơn (**37.238** so với **34.819 ms**). Nếu đổi trục chất lượng sang F1 test 775 và bỏ LLM, đường Pareto tính từ bảng là **1, 2, 3b, 3c**; khi ấy 3c lấn át 3a. Cần nói rõ tập đánh giá trước khi kết luận.

SUB chỉ có 20 spam; từ dự đoán hoàn hảo, một FN làm F1 giảm từ 1 xuống `38/39 ≈ 0.9744`, một FP làm giảm xuống `40/41 ≈ 0.9756`. Khoảng cách khoảng 0.025 có thể chỉ đến từ một tin, nên chênh lệch nhỏ chưa đủ để xếp hạng chắc chắn.

## 4. Phân tích 50 case khó

### Phạm vi bằng chứng stress test

Notebook có bộ gốc **50 case**, nhưng bảng tổng hợp lưu gần nhất cho accuracy **0.509, 0.527, 0.691, 0.545, 0.600, 0.436**. Các giá trị này khớp lần lượt **28/55, 29/55, 38/55, 30/55, 33/55, 24/55** khi làm tròn, phù hợp với phần nhận xét đã thêm 5 case. Trong khi đó, cell `MY_CASES` hiện có cả 5 dòng bị chú thích, và output đọc lỗi lại ghi `29/50`. Vì vậy, các bảng dưới đây là **output stress test đã lưu, có dấu hiệu dùng 55 case**, chưa phải kết quả 50 case gốc được tái lập đồng nhất.

| Phương án | F1 test 775 | F1 stress test đã lưu | Mức giảm F1 | Accuracy stress test đã lưu |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.8969 | 0.308 | 0.589 | 50.9% |
| BoW + MLP | 0.9458 | 0.381 | 0.565 | 52.7% |
| MiniLM + kNN | 0.9493 | **0.721** | 0.228 | **69.1%** |
| MiniLM + LogReg | 0.9488 | 0.510 | 0.439 | 54.5% |
| MiniLM + MLP head | 0.9665 | 0.593 | 0.374 | 60.0% |
| LLM nhỏ zero-shot | Chưa đo | 0 | Không so được trên test 775 | 43.6% |
| LLM nhỏ + prompt phòng thủ | Chưa đo | 0 | Không so được trên test 775 | 43.6% |

Trong các mô hình cục bộ, **BoW + LogReg tụt nhiều nhất**, khoảng 0.589 F1; kNN giữ kết quả tốt nhất trên output khó đã lưu. Test thường cùng nguồn với train và vẫn có thể chứa các biến thể gần trùng, nên không bao phủ đầy đủ kiểu spam mới, mạo danh và né từ khóa. Bộ khó cố ý tập trung vào các bẫy, phù hợp để tìm điểm yếu hơn là ước lượng tỷ lệ lỗi SMS hằng ngày.

### Nhóm case nào khó nhất?

Tỷ lệ đúng dưới đây lấy trực tiếp từ bảng HTML theo nhóm trong notebook, làm tròn đến phần trăm; cùng chịu giới hạn về số case đã nêu.

| Nhóm | BoW + LR | BoW + MLP | Emb + kNN | Emb + LR | Emb + MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ham trông như spam | 100% | 91% | 73% | 82% | 82% |
| Thao túng / lừa đảo | 11% | 22% | 89% | 22% | 33% |
| Spam giọng tự nhiên | **10%** | **10%** | 40% | **10%** | **30%** |
| Né từ khóa | 14% | 14% | 86% | 71% | 71% |
| Prompt injection | 67% | 83% | 83% | 100% | 100% |
| Nói về spam | 100% | 100% | **40%** | 40% | 40% |
| Tin giao dịch hợp lệ | 71% | 71% | 71% | 71% | 71% |

BoW bị ảnh hưởng mạnh bởi né từ khóa, nhưng **nhóm thấp nhất thực tế là spam giọng tự nhiên (10%)**, khác dự đoán ban đầu. Với kNN, spam giọng tự nhiên và nói về spam cùng thấp nhất (40%); với hai head embedding, spam giọng tự nhiên thấp nhất. Không thể gộp mọi mô hình embedding vào một kết luận chung rằng ham trông như spam là nhóm khó nhất.

### Hai case cụ thể và cơ chế sai

- **Case #14 — spam giọng tự nhiên:** “Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back”. Output đọc lỗi ghi **tất cả mô hình đều sai**. Câu kể chuyện cá nhân và lời mời nhẹ nhàng che đi mồi nhử thu nhập; các mẫu train thiên về spam có cấu trúc từ khóa rõ hơn nên mô hình có thể bỏ lọt. Cả BoW lẫn encoder đều chưa đủ để suy ra chắc chắn ý đồ từ giọng điệu thân thiện.
- **Case #16 — giả làm bạn cũ:** “Long time no see! I've started a little side business with skincare, you'd love it, can I send you the catalogue link?”. Đây cũng là case **tất cả mô hình đều sai** trong output đọc lỗi. Lời chào thân mật và câu xin phép gửi catalogue gần với trò chuyện thường, trong khi nhãn của bộ khó quy định đây là spam bán hàng. Cơ chế đáng chú ý là thiếu tín hiệu về quan hệ người gửi/người nhận và việc người nhận có đồng ý nhận quảng cáo hay không.

Ngoài ra, `C0ngr4tulat10ns` hoặc `F R E E` làm token lệch khỏi vocab của BoW. Có thể thử chuẩn hóa chữ/số và ký tự chèn, nhưng phải kiểm tra lại trên validation vì chuyển `0 → o` hoặc gộp ký tự bừa bãi có thể làm sai mã giao dịch và tăng chặn nhầm ham.

**Prompt injection:** 4a và 4d đều F1 **0**, accuracy **43.6%** trong bảng lưu. Kết quả này chưa chứng minh prompt phòng thủ có cải thiện hoặc LLM đã làm theo lệnh tấn công, vì luồng đầu ra LLM vốn đã không hợp lệ trên SUB. Tách SMS vào thẻ và coi nội dung là dữ liệu là hướng phòng thủ của notebook; cần kiểm tra phản hồi hợp lệ trên cùng các case trước khi đánh giá hiệu quả.

**Case tự nghĩ:** notebook còn 5 ý tưởng dạng chú thích thuộc các nhóm ham trông như spam, né từ khóa, nói về spam, mạo danh và giao dịch hợp lệ. Do `MY_CASES` hiện rỗng và chưa có kết quả riêng cho từng case, báo cáo chưa xác nhận hoàn thành bonus 5 case hoặc quy kết cụ thể model nào bị từng case đánh lừa.

## 5. Thử thách cá nhân

### Đề và cách làm trong notebook

SEED 3413 nhận **đề #6**: yêu cầu LLM trả JSON `{"label": ..., "reason": ...}`, viết parser, chạy 10 tin SUB và so token/chi phí với bản một từ.

Cell thử thách đã có `build_json_prompt`, yêu cầu hai trường `label`, `reason`, giải thích tiếng Việt ngắn và không bọc Markdown. `parse_json_label` bỏ dấu bọc Markdown nếu có, dùng `json.loads`, rồi chuyển nhãn sang 0/1. Thí nghiệm lấy **10 tin đầu của SUB** và tăng giới hạn đầu ra lên **150 token** để chứa JSON và lý do. Phần này chỉ tổng hợp code và output sẵn có, không bổ sung lời giải code cho thử thách.

### Kết quả và chi phí

Output lưu **10/10 phản hồi JSON parse được và nhãn đúng**: **7 spam, 3 ham**, suy ra accuracy/F1 trên 10 tin đều bằng **1.0**. Đây là một tập rất nhỏ, đồng thời đã đổi cả prompt và giới hạn token, nên chưa đủ để kết luận JSON làm năng lực phân loại tăng so với bản một từ.

| Đại lượng | Bản một từ | Bản JSON | Cách hiểu |
| --- | ---: | ---: | --- |
| Token trung bình/tin | 101 | 135.5 | Bản một từ dùng giả định; JSON ước lượng bằng số từ chia 0.75, tăng khoảng 34.1%. |
| USD/1.000 tin | 0.01050 | 0.03900 | Tính theo giả định riêng: một từ có 100 input + 1 output; JSON có 140 input + 50 output. |

Chi phí JSON theo giả định là `(140*0.10 + 50*0.50)/1000 = 0.03900 USD/1.000 tin`, gấp khoảng **3.71 lần** bản một từ. Con số **135.5 token/tin** trong output và giả định **190 token/tin** dùng tính tiền không cùng cơ sở; vì vậy chi phí trên chỉ là minh họa, chưa phải chi phí thực đo từ API `usage`. Phần trả lời cũ ghi **116.1 token/tin** không khớp output 135.5, nên báo cáo dùng output và nêu giới hạn ước lượng.

JSON hữu ích khi cần lưu lý do cho kiểm tra hoặc hỗ trợ người dùng; lọc tự động quy mô lớn cần cân nhắc chi phí và độ trễ thêm. Lý do LLM sinh ra cũng cần được kiểm chứng, không tự động bảo đảm quyết định đúng. Parser hiện kiểm tra chuỗi con `"spam"` trước `"ham"`, nên vẫn có rủi ro nhận sai `"not spam"`; cần đối chiếu nhãn đúng tập giá trị cho phép khi tự hoàn thiện lại bài.

## 6. Phương án cho sản phẩm thật

### Lựa chọn theo chất lượng và SLA

Với yêu cầu **p95 < 50 ms/tin**, phương án **3c — MiniLM + MLP head** là ứng viên ưu tiên để đánh giá triển khai: F1 test **0.9665**, precision **0.9806**, recall **0.9528**, p95 **37.238 ms**. Trên test, mô hình có **TP=101, FP=2, FN=5, TN=667**. So với BoW + MLP, số spam bỏ lọt giảm từ **10 xuống 5**, đổi lại p95 tăng từ **0.265 lên 37.238 ms**.

Các phương án 1, 2, 3b và 3c đều đạt SLA trong phép đo đã lưu. kNN vượt SLA ở **64.005 ms**; LLM ở khoảng **1.410–1.635 giây** và còn lỗi đọc nhãn. Tuy vậy, 3c chỉ đạt F1 **0.593** trên stress test đã lưu, nên cần bổ sung dữ liệu khó và đánh giá lại trước khi dùng làm bộ lọc tự động cho toàn bộ lưu lượng.

### Quy mô 10 triệu SMS/ngày

Lưu lượng trung bình là `10.000.000/86.400 ≈ 115.74 tin/s`. Throughput cả lô của 3c đo được khoảng **43.71 tin/s**, nên phép chia đơn giản cho thấy cần ít nhất **3 đơn vị xử lý tương đương phép đo** để phục vụ tải trung bình. Đây chỉ là ước lượng công suất, chưa bảo đảm SLA khi có xếp hàng hoặc tải đỉnh; cần đo đồng thời, dự phòng công suất và kiểm thử p95 toàn hệ thống. Không suy throughput bằng cách lấy nghịch đảo p95 vì hai chỉ số được đo theo cách khác nhau.

Nếu phần cứng hạn chế, **BoW + MLP** là phương án thay thế nhanh: F1 **0.9458**, khoảng **44.855 tin/s** cả lô, p95 **0.265 ms**, nhưng bỏ lọt spam nhiều hơn và F1 stress test chỉ **0.381**. Chọn 3c nghĩa là chấp nhận encoder nặng hơn để đổi lấy chất lượng test tốt hơn; chọn BoW nghĩa là ưu tiên công suất và ngân sách phần cứng.

### Chi phí và hướng kết hợp

Theo giả định tính tay **100 token input + 1 token output**, giá cấu hình lab cho model nhỏ cho chi phí **105 USD/ngày** ở 10 triệu tin, khoảng **3.150 USD/30 ngày**. Nếu chỉ nhân đơn giá ước lượng từ lượt zero-shot hiện có, số tương ứng khoảng **64.97 USD/ngày**; nhưng lượt đó không tạo được nhãn hợp lệ nên không phải chi phí của một dịch vụ lọc hoạt động đúng. Cả hai phép tính chưa gồm retry, hạ tầng và vận hành.

Có thể thử chuyển các tin chưa chắc chắn sang LLM sau khi sửa luồng đầu ra, hoặc cho xử lý bổ sung bất đồng bộ. Ngưỡng chuyển tuyến phải chọn bằng validation và đo tỷ lệ chuyển thực tế. Với nhánh LLM trễ trên một giây, cần đo lại p95 của toàn hệ thống; chưa có bằng chứng rằng router giữ được SLA, phục vụ ngay hơn 90% tin hoặc chống prompt injection hoàn toàn.

## 7. AI log và điều đã học

### AI log

| Lần sử dụng có bằng chứng | Đã hỏi | Hỗ trợ nhận được | Thay đổi thực hiện |
| --- | --- | --- | --- |
| Phiên biên soạn báo cáo ngày 06/11/2026 | Đọc `day01_lab` và các file liên quan để hoàn thiện báo cáo cá nhân | Codex đối chiếu notebook, JSON, ảnh tính tay, rubric và hướng dẫn nộp; tổng hợp bảng kết quả, kiểm tra dấu thời gian và chỉ ra các nhận xét chưa khớp output | Điền báo cáo Markdown này, ghi rõ phần phân tích được AI hỗ trợ, các số ước lượng và các bằng chứng chưa đồng nhất; không sửa code notebook, không tạo lại log dự đoán hoặc chạy API. |

Các tài liệu hiện có không chứa nhật ký những lần dùng AI trước phiên này, nên chưa thể xác nhận hoặc dựng lại lịch sử đó. Nếu đã dùng AI khi làm notebook, người nộp cần bổ sung đúng câu hỏi, hỗ trợ nhận được và phần tự sửa tương ứng; không thể suy ra “không dùng AI” chỉ từ việc thiếu log.

### Điều rút ra từ kết quả

Accuracy cao chưa bảo đảm bắt được spam tốt: LogReg đạt 97.42% accuracy nhưng bỏ lọt 19/106 spam, trong khi bộ khó làm F1 giảm còn 0.308. Một luồng LLM trả nhãn không hợp lệ rồi mặc định thành ham có thể trông như lỗi mô hình, nên cần kiểm tra đầu ra và parser trước khi xếp hạng model. Việc chọn phương án phải đồng thời dựa vào tập đánh giá, độ bền trước case mới, p95 và công suất thực tế, thay vì chỉ nhìn F1 cao nhất trên SUB 40.
