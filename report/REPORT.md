# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

  ```json
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
  ```
- Record này mô tả toàn ảnh như thế nào? Đây là một nhãn cho cả ảnh, không có tọa độ nên không chỉ ra chiếc xe
  nào là "cab". Ảnh `traffic` là một con đường đông đúc, nổi bật nhất là nhiều xe buýt, nên `cab` (score 0.51)
  chỉ đúng một phần cảnh vật. Ở ảnh `kitchen`, hạng 1 là `gong` (0.42): mô hình có vẻ nhầm nồi chảo treo tường
  thành cồng chiêng.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Taxonomy của tập dữ liệu dùng để huấn luyện checkpoint
  (ImageNet-1K, 1000 lớp), không phải mô hình. Mô hình chỉ xếp hạng trong 1000 lớp này. Vì ImageNet-1K không có
  lớp "cảnh giao thông" hay "nhà bếp", mô hình phải chọn lớp gần nhất.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Cùng một khái niệm có ID và tên khác nhau ở mỗi taxonomy. Ví dụ
  bàn ăn là `532 "dining_table"` trong ImageNet-1K (`classification_predictions.json`, sample `dining`) nhưng lại là
  `60 "dining table"` trong COCO-80 (`detection_predictions.json`, sample `dining`). Chỉ có ID thì không biết số đó
  thuộc bảng nào; chỉ có tên thì dễ nhầm vì khác biệt nhỏ về cách viết.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Cần quy định single-label hay multi-label. Nếu là
  single-label thì chọn chủ thể theo tiêu chí nào (diện tích lớn nhất, tiền cảnh hay mục đích dự án), có lớp
  "cảnh/không xác định" không, và ca hai lớp ngang nhau thì escalation cho ai. Ví dụ ảnh `traffic` chủ yếu là
  xe buýt chứ không phải taxi.
- Vì sao model score không phải ground truth? Score chỉ là mức tự tin của mô hình, dùng để xếp hạng; chưa có ai
  xác nhận theo guideline. Score cao cũng có thể sai: `restaurant` được 0.79 cho ảnh `dining`, trong khi ảnh
  chỉ là một bàn ăn. Ground truth phải do annotator tạo theo guideline và reviewer kiểm tra.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

  ```json
  {"sample_id": "kitchen", "class_name": "person", "score": 0.912625,
   "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}
  ```
- Diễn giải vị trí box bằng lời: Gốc (0, 0) là góc trên trái ảnh 640×427. Góc trên trái của box ở (x≈385, y≈69),
  góc dưới phải ở (x≈499, y≈349), tức rộng 113.58 px và cao 279.68 px. Box nằm ở nửa phải ảnh và kéo dài từ gần
  mép trên xuống khoảng 82% chiều cao; đó là người đầu bếp đứng quay lưng.
- So sánh số prediction ở hai threshold: Ở `kitchen`, threshold 0.35 cho 11 box (person ×2, oven ×2, bowl ×5,
  cup ×2). Threshold 0.60 (lọc các record có score ≥ 0.60) còn 6 box. Năm box bị loại là bowl 0.50, bowl 0.46,
  cup 0.45, cup 0.38, bowl 0.38.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp bao phủ nhiều object thật hơn
  (cốc, bát nhỏ) nhưng cũng giữ nhiều box đáng ngờ, như hai box `cup` chồng nhau; reviewer phải xem và loại nhiều
  hơn. Threshold cao cho ít box hơn nhưng bỏ sót nhiều hơn (hai cốc thật bị loại, trong khi `person 0.61` chỉ là
  một cánh tay ở mép vẫn được giữ), nên annotator phải vẽ bù. Threshold chỉ là cấu hình lọc prediction, không
  phải quy tắc bỏ qua object khi tạo ground truth.
- Đề xuất một quy tắc box chặt: Box ôm sát phần nhìn thấy được của object, bốn cạnh chạm pixel ngoài cùng với sai
  lệch ≤ 2 px. Không chứa bóng đổ hay object khác, không cắt mất phần nhìn thấy. Mỗi object đúng một box, không
  vẽ box trùng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quyết định box
  chỉ bao phần nhìn thấy hay ước lượng cả phần bị che, cần nhìn thấy tối thiểu bao nhiêu thì mới gán nhãn (ví dụ
  `person 0.61` ở mép trái chỉ thấy một cánh tay), và có gắn cờ `truncated`/`occluded` không. Ca lớp mơ hồ như
  `oven 0.63` bên phải (`[489.09, 200.73, 617.81, 344.88]`), trông giống bồn rửa (`sink`), cần escalation; annotator
  không tự quyết.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

  ```json
  {"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318,
   "polygon_point_count": 348,
   "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], "…"]}
  ```
- Polygon bổ sung chi tiết gì so với box? Box chỉ có 4 số và chứa cả tường, nồi phía sau người. Polygon có 348
  điểm pixel bám theo viền đầu, vai, tay, tạp dề và chân, nên loại được nền (kể cả khoảng trống giữa hai chân).
  Polygon cho biết pixel nào thuộc object, còn box chỉ cho biết object nằm trong vùng chữ nhật nào.
- `instance_id` dùng để làm gì và không phải loại ID nào? Dùng để phân biệt từng object trong output của bài lab.
  Ví dụ `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` là 4 `bowl` khác nhau, cùng `class_id=45`
  nhưng mỗi cái có polygon riêng. `instance_id` không phải class ID và không phải tracking ID (không theo dõi
  object qua nhiều khung hình).
- Đề xuất một quy tắc biên mask: Polygon bám theo biên nhìn thấy được với sai lệch ≤ 2 px, không bao nền hay object
  đè lên, loại các lỗ thật (khoảng trống giữa tay và thân). Mỗi pixel thuộc tối đa một instance.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Trên ảnh, mask `dining table`
  (`kitchen-005`) phủ luôn bột và đĩa bánh trên bàn, mask `oven` (`kitchen-007`) chồng lên bàn và bát. Guideline
  cần quyết định vật đặt trên bàn có bị cắt khỏi mask bàn không, ranh giới tại vùng hai object tiếp xúc đặt ở đâu,
  và biên mờ (tóc, vùng tối) cho phép dung sai bao nhiêu. Ca lớp sai hoặc mơ hồ, như `potted plant 0.63`
  (`kitchen-004`) thực ra là bó thảo mộc treo, cần escalation.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                     | Lỗi hoặc điểm mơ hồ quan sát được                                                                                | Annotator làm gì?                                                                                                  | Reviewer xem gì?                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | 1 nhãn cho mỗi ảnh:`taxonomy_name` + `class_id` + `class_name` | `traffic` → `cab` dù ảnh chủ yếu là xe buýt; `kitchen` → `gong` (sai)                                      | Chọn lớp theo tiêu chí chủ thể trong guideline, không chép prediction; gắn cờ ca nhiều chủ thể          | Nhãn có đúng tiêu chí không; ID, tên và taxonomy có khớp không; ca mơ hồ lặp lại thì đề xuất sửa guideline           |
| Phát hiện vật thể | 1 box`xyxy` pixel + 1 lớp COCO-80 cho mỗi object                    | `oven 0.63` có thể là `sink`; `person 0.61` chỉ là cánh tay; bỏ sót lọ mứt, đĩa bánh                    | Vẽ box chặt cho mọi object thuộc taxonomy, kể cả object mô hình bỏ sót; gắn cờ`truncated`/`occluded` | Có thiếu object hoặc box trùng không; box có chặt không; lớp đúng chưa; ca mơ hồ đã escalation chưa                      |
| Instance segmentation | 1 polygon pixel + 1 lớp + 1`instance_id` cho mỗi instance           | `potted plant` thực ra là bó thảo mộc; mask bàn phủ cả đồ vật trên bàn; mask `oven` chồng lên bàn/bát | Vẽ polygon theo biên nhìn thấy, loại lỗ và vật đè lên, tách riêng các instance cùng lớp              | Biên có sát không; có chồng lấn hay trùng instance không;`instance_id` có duy nhất không; mask lỗi thì trả về làm lại |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng dữ liệu trong phạm vi được phê duyệt: 3 ảnh COCO công khai (CC BY 2.0,
  ghi nguồn trong `IMAGE_ATTRIBUTION.md`) mà notebook tải và kiểm tra checksum. Không tải ảnh cá nhân, khuôn mặt,
  biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai; tải ảnh lỗi thì báo mentor, không
  thay bằng ảnh khác. Họ tên/MSSV chỉ nằm trong tên repository, không ghi vào `REPORT.md`, JSON, PNG hay ZIP.
  Ảnh `traffic` có biển số và người đi bộ: trong bài lab được phép vì là dữ liệu công khai có license, còn trong
  dự án thật phải làm mờ hoặc hạn chế truy cập theo chính sách.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach/GV phụ trách lớp. Tôi không
  tiếp tục xử lý, không tự chia sẻ hay tự sửa dữ liệu đó; nếu lỡ đã push lên GitHub, tôi báo ngay GV/Lab Coach
  để xử lý.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
