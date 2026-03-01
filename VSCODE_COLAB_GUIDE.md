# 🔗 Kết Nối VS Code với Google Colab GPU

## ✅ Bạn Đã Có Extension: `google.colab`

Có thể chạy notebook trong VS Code và dùng GPU của Colab!

---

## 🚀 Cách Sử Dụng (3 Bước Đơn Giản)

### **Bước 1: Chọn Kernel Colab**

1. Mở file `train_notebook.ipynb` trong VS Code (đang mở rồi ✅)
2. Nhìn lên **góc trên bên phải**, click vào nút kernel (hiện đang là "Python 3" hoặc tương tự)
3. Hoặc nhấn: `Ctrl + Shift + P` → gõ "**Select Notebook Kernel**"

Sẽ hiện menu:
```
┌─────────────────────────────────────────┐
│  Select Kernel                          │
├─────────────────────────────────────────┤
│  ○ Python Environments (Local)          │
│  ○ Jupyter Server                       │
│  ● Connect to Google Colab Runtime   ← CHỌN CÁI NÀY
└─────────────────────────────────────────┘
```

### **Bước 2: Đăng Nhập Google**

- Browser sẽ tự động mở
- Đăng nhập tài khoản Google
- Click "Allow" để cho phép VS Code truy cập Colab
- Quay lại VS Code

Sẽ thấy kernel đổi thành: **"Colab Runtime (T4 GPU 16GB)"** ✅

### **Bước 3: Chạy Notebook**

Click **Run All** hoặc chạy từng cell như bình thường!

Code chạy trên GPU Colab, nhưng bạn edit trong VS Code!

---

## 🎯 Kiểm Tra Nhanh

Chạy cell này để xem đang dùng GPU nào:

```python
import torch
print(f"🔥 CUDA: {torch.cuda.is_available()}")
print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Kiểm tra có phải Colab không
import sys
if 'google.colab' in sys.modules:
    print("✅ Đang chạy trên Google Colab Runtime!")
else:
    print("💻 Đang chạy local")
```

Nếu thấy "NVIDIA Tesla T4" hoặc "V100" → Thành công! 🎉

---

## 📋 Workflow Hoàn Chỉnh

```
┌──────────────────────────────────────────────────────────┐
│  1. Viết code trong VS Code (máy bạn)                    │
│     ↓                                                     │
│  2. Click "Run Cell"                                     │
│     ↓                                                     │
│  3. Code gửi lên Google Colab                           │
│     ↓                                                     │
│  4. Chạy trên GPU T4 (16GB VRAM)                        │
│     ↓                                                     │
│  5. Kết quả hiển thị trong VS Code                      │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup Lần Đầu cho Colab Runtime

Khi kết nối Colab lần đầu, thêm cell này vào đầu notebook:

```python
# Cell 0: Setup cho Colab (chỉ chạy lần đầu)
import sys
import os

if 'google.colab' in sys.modules:
    print("🌐 Setting up Colab environment...")
    
    # Cài dependencies
    !pip install -q --upgrade transformers datasets accelerate tensorboard
    
    # Upload project files
    print("📤 Upload file transformer-project.zip ở bước tiếp theo")
    from google.colab import files
    uploaded = files.upload()
    
    # Extract
    !unzip -q transformer-project.zip
    %cd transformer-project
    
    print("✅ Setup done! Bây giờ chạy các cell còn lại")
```

**Lưu ý:** Cần nén dự án thành ZIP trước:
```powershell
# Trong terminal VS Code
cd transformer-project
Compress-Archive -Path * -DestinationPath ../transformer-project.zip
```

---

## 🔄 So Sánh 3 Cách

| Phương án | Edit Code | Chạy Ở Đâu | GPU | Độ Khó |
|---|---|---|---|---|
| **1. Local (hiện tại)** | VS Code | Máy bạn | Máy bạn | ⭐ Dễ |
| **2. VS Colab (kernel)** ⭐ | VS Code | Colab | T4 (16GB) | ⭐⭐ Trung bình |
| **3. Upload lên Colab** | Web Colab | Colab | T4 (16GB) | ⭐ Dễ |

**Khuyến nghị:** Dùng **Cách 2** (kết nối kernel) - giữ workflow VS Code nhưng dùng GPU Colab!

---

## 🐛 Troubleshooting

### Không thấy "Connect to Google Colab" trong kernel menu?

**Giải pháp:**
1. Kiểm tra extension đã bật:
```powershell
code --list-extensions | Select-String colab
```

2. Cài lại extension nếu cần:
```powershell
code --install-extension google.colab
```

3. Reload VS Code: `Ctrl+Shift+P` → "Reload Window"

### Lỗi "Failed to connect to Colab"?

**Giải pháp:**
- Kiểm tra internet
- Đăng nhập lại Google
- Thử disconnect và connect lại kernel

### Upload file quá lâu?

**Giải pháp:**
- Chỉ upload các file cần thiết (bỏ `outputs/`, `logs/`)
- Hoặc dùng Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/transformer-project
```

### Colab bị disconnect sau 1 tiếng?

**Giải pháp:**
- Colab miễn phí timeout sau 12 giờ idle
- Nâng cấp Colab Pro ($10/tháng) để giữ kết nối lâu hơn
- Hoặc chạy cell này để keep alive:
```python
import time
while True:
    time.sleep(60)  # Ping mỗi phút
```

---

## 💡 Tips & Tricks

### 1. Lưu Checkpoint Tự Động

Thêm vào config:
```yaml
training:
  save_steps: 100
  save_total_limit: 3
```

### 2. Sync với Google Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoint
!cp -r outputs/ /content/drive/MyDrive/checkpoints/
```

### 3. Monitor Training

```python
# TensorBoard trong VS Code
%load_ext tensorboard
%tensorboard --logdir logs/tensorboard
```

### 4. Download Kết Quả

```python
# Nén và download
!zip -r results.zip outputs/
from google.colab import files
files.download('results.zip')
```

---

## 📊 So Sánh Hiệu Suất

| Thiết bị | Training Time (1 epoch) | VRAM | Chi phí |
|---|---|---|---|
| Colab T4 | 15 phút | 16GB | Miễn phí |
| Colab V100 (Pro) | 8 phút | 16GB | $10/tháng |
| Colab A100 (Pro+) | 4 phút | 40GB | $50/tháng |
| Local RTX 3060 | 20 phút | 12GB | Đã mua rồi |
| Local RTX 4090 | 5 phút | 24GB | $1600 |

**Kết luận:** Colab T4 miễn phí là đủ tốt cho training PhoBERT!

---

## ✅ Checklist

- [ ] Extension `google.colab` đã cài (đã có ✅)
- [ ] Đã nén project thành ZIP
- [ ] Click "Select Kernel" → "Connect to Colab"
- [ ] Đăng nhập Google
- [ ] Chạy cell setup (upload ZIP)
- [ ] Chạy training và enjoy! 🎉

---

## 🎓 Tóm Tắt

**3 bước:**
1. **Select Kernel** → Connect to Google Colab
2. **Đăng nhập** Google
3. **Chạy notebook** như bình thường

**Lợi ích:**
✅ Edit trong VS Code (quen thuộc)  
✅ Chạy trên GPU Colab (mạnh + miễn phí)  
✅ Không phải paste code qua lại  
✅ TensorBoard, debugging thoải mái  

**Nhược điểm:**
⚠️ Cần internet tốt  
⚠️ Timeout sau 12 giờ (miễn phí)  
⚠️ Setup lần đầu hơi phức tạp  

---

**Thử ngay:** Click nút kernel ở trên và chọn "Connect to Google Colab"! 🚀
