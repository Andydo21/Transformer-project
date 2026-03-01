# 🌐 Hướng dẫn sử dụng Web UI

## 🚀 Khởi động Server

### Bước 1: Cài đặt dependencies (nếu chưa có)
```bash
cd transformer-project
pip install -r requirements.txt
```

### Bước 2: Chạy server
```bash
cd scripts
python serve_api.py
```

Server sẽ chạy tại: **http://localhost:8000**

---

## 🎯 Các tính năng

### 1. 📋 Phân loại văn bản (Classification)
- **Mục đích**: Xác định loại văn bản pháp lý
- **8 loại**: Hình sự, Dân sự, Hành chính, Giao thông, Doanh nghiệp, Đất đai, Lao động, Bất động sản
- **Cách dùng**:
  1. Nhập văn bản vào ô text
  2. Click "🚀 Phân tích"
  3. Xem kết quả với độ tin cậy và xác suất chi tiết

### 2. 🏷️ Nhận diện thực thể (NER)
- **Mục đích**: Trích xuất các thực thể trong văn bản
- **Thực thể**: Luật, Điều khoản, Tổ chức, Người, Ngày tháng, Địa điểm
- **Cách dùng**:
  1. Nhập văn bản pháp lý
  2. Click "🔍 Trích xuất thực thể"
  3. Xem các thực thể được phân loại theo màu

### 3. ❓ Hỏi đáp (Question Answering)
- **Mục đích**: Trả lời câu hỏi dựa trên văn bản
- **Cách dùng**:
  1. Nhập văn bản pháp lý vào "Ngữ cảnh"
  2. Đặt câu hỏi về nội dung văn bản
  3. Click "💬 Tìm câu trả lời"
  4. Xem câu trả lời với độ tin cậy

**Ví dụ**:
- Ngữ cảnh: "Theo Điều 10 Luật Lao động, người lao động có quyền nghỉ phép 12 ngày mỗi năm..."
- Câu hỏi: "Người lao động được nghỉ phép bao nhiêu ngày?"
- Trả lời: "12 ngày mỗi năm"

### 4. 📝 Tóm tắt văn bản (Summarization)
- **Mục đích**: Tạo bản tóm tắt ngắn gọn
- **Cách dùng**:
  1. Nhập văn bản pháp lý dài
  2. Click "📄 Tóm tắt"
  3. Xem bản tóm tắt ngắn gọn

---

## 🔧 Cấu hình nâng cao

### Load model tùy chỉnh
```bash
# Set model path trước khi chạy
export MODEL_PATH="outputs/classification/best_model.pt"
python serve_api.py
```

### Đổi port
```bash
export PORT=8080
python serve_api.py
```

---

## 🌐 API Endpoints

### Web UI
- **GET /** - Giao diện web chính

### REST API
- **GET /api** - Thông tin API
- **GET /health** - Health check
- **POST /predict** - Phân loại văn bản
- **POST /ner** - Nhận diện thực thể
- **POST /qa** - Hỏi đáp
- **POST /summarize** - Tóm tắt văn bản
- **POST /batch-predict** - Phân loại nhiều văn bản

### API Documentation
- **GET /docs** - Swagger UI (auto-generated)
- **GET /redoc** - ReDoc documentation

---

## 🎨 Giao diện

### Features:
- ✅ **Modern UI** với gradient tím đẹp mắt
- ✅ **Responsive** - tự động điều chỉnh theo màn hình
- ✅ **Tab switching** - chuyển đổi giữa các chức năng
- ✅ **Real-time status** - hiển thị trạng thái API
- ✅ **Loading animations** - hiệu ứng đang xử lý
- ✅ **Error handling** - thông báo lỗi rõ ràng
- ✅ **Color-coded results** - kết quả phân loại theo màu
- ✅ **Confidence bars** - thanh hiển thị độ tin cậy
- ✅ **Probability charts** - biểu đồ xác suất chi tiết

---

## 🔍 Test với curl (cho developers)

### Phân loại
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hợp đồng mua bán nhà đất tại Hà Nội",
    "return_probabilities": true
  }'
```

### NER
```bash
curl -X POST http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Theo Điều 10 Luật Lao động năm 2019..."
  }'
```

### QA
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Theo Luật Lao động, người lao động được nghỉ phép 12 ngày mỗi năm.",
    "question": "Người lao động được nghỉ bao nhiêu ngày?"
  }'
```

### Tóm tắt
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Văn bản pháp lý dài...",
    "max_length": 150,
    "min_length": 50
  }'
```

---

## ⚠️ Lưu ý

1. **Model phải được train trước**: Server sẽ load model từ `MODEL_PATH`
2. **Mỗi endpoint yêu cầu model tương ứng**:
   - `/predict` → Classification model
   - `/ner` → NER model
   - `/qa` → QA model
   - `/summarize` → Summarization model
3. **Health check**: Kiểm tra `/health` để xem model đã load chưa
4. **CORS enabled**: API có thể gọi từ bất kỳ origin nào

---

## 🐛 Troubleshooting

### Server không start
```bash
# Check port đã được dùng chưa
netstat -ano | findstr :8000

# Dùng port khác
export PORT=8080
python serve_api.py
```

### Model không load
```bash
# Check MODEL_PATH
echo $MODEL_PATH

# Set đúng path
export MODEL_PATH="outputs/classification/best_model.pt"
```

### API trả về 503
- Model chưa được load
- Check logs để xem lỗi gì
- Chắc chắn model file tồn tại

### Web UI không hiện
- Check folder `templates/` và `static/` tồn tại
- Check file `index.html` có trong `templates/`

---

## 📊 Screenshots

### Phân loại văn bản
```
┌─────────────────────────────────────┐
│ 📋 Phân loại văn bản pháp lý        │
├─────────────────────────────────────┤
│ [Nhập văn bản...]                   │
│                                     │
│ 🚀 Phân tích                        │
│                                     │
│ ✨ Kết quả:                         │
│ Loại: Bất động sản                  │
│ ████████████████░░░░ 89.5%         │
└─────────────────────────────────────┘
```

---

## 📝 Next Steps

Sau khi chạy web UI:
1. Test từng chức năng
2. Upload file PDF/DOCX (nếu thêm tính năng upload)
3. Integrate vào hệ thống lớn hơn
4. Deploy lên production server

**Enjoy! 🚀**
