# 🌐 Hướng Dẫn Chạy trên Google Colab

## 📌 TL;DR
Notebook hiện tại chạy **LOCAL** (máy bạn). Để dùng GPU Colab, làm theo:

---

## Cách 1: Upload Code Lên Colab (Khuyến nghị ⭐)

### Bước 1: Nén Project
```powershell
# Trong thư mục transformer-project
Compress-Archive -Path * -DestinationPath ../transformer-project.zip
```

### Bước 2: Tạo Colab Notebook Mới
1. Vào https://colab.research.google.com/
2. File → New Notebook
3. Runtime → Change runtime type → **GPU (T4 hoặc V100)**

### Bước 3: Upload & Extract trong Colab
```python
# Cell 1: Upload file zip
from google.colab import files
uploaded = files.upload()  # Click chọn transformer-project.zip

# Cell 2: Extract
!unzip -q transformer-project.zip -d transformer-project
%cd transformer-project
!ls -la

# Cell 3: Install dependencies
!pip install -q transformers torch datasets accelerate tensorboard

# Cell 4: Kiểm tra GPU
import torch
print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Cell 5-10: Copy các cell từ train_notebook.ipynb
# Hoặc chạy trực tiếp:
!python main.py train --config configs/advanced_config.yaml
```

### Ưu điểm:
✅ Miễn phí GPU T4 (16GB VRAM)  
✅ Không cần cài đặt gì  
✅ Chạy trên cloud  

### Nhược điểm:
❌ Timeout sau 12 giờ  
❌ Mất data khi disconnect  
❌ Phải upload lại mỗi lần  

---

## Cách 2: Kết Nối Colab với Google Drive

### Bước 1: Mount Google Drive trong Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy project vào Drive trước
%cd /content/drive/MyDrive/transformer-project
```

### Bước 2: Chạy như bình thường
```python
!python main.py train --config configs/advanced_config.yaml
```

### Ưu điểm:
✅ Không mất data  
✅ Sync tự động với Drive  

### Nhược điểm:
❌ Cần upload project lên Drive  
❌ Phải mount mỗi lần  

---

## Cách 3: VS Code Remote - Colab (Phức tạp ⚠️)

### Yêu cầu:
- Extension: **Colab** (by Google)
- Tài khoản Google

### Bước 1: Cài Extension
```powershell
code --install-extension ms-python.vscode-pylance
```

### Bước 2: Kết nối trong VS Code
1. Ctrl+Shift+P → "Colab: Connect to Colab"
2. Chọn runtime (GPU)
3. Chạy notebook như bình thường

### Ưu điểm:
✅ Giữ nguyên workflow VS Code  
✅ Edit local, chạy remote  

### Nhược điểm:
❌ Setup phức tạp  
❌ Cần internet tốt  
❌ Extension còn beta  

---

## 🎯 Khuyến Nghị

| Tình huống | Dùng cách nào |
|---|---|
| Chạy nhanh 1 lần | **Cách 1** (Upload) |
| Training lâu dài | **Cách 2** (Drive) |
| Muốn giữ workflow VS Code | **Cách 3** (Remote) |
| Có GPU máy tính | **Chạy local** (không cần Colab) |

---

## ⚡ So Sánh GPU

| GPU | VRAM | Tốc độ | Nơi chạy |
|---|---|---|---|
| Colab T4 | 16GB | 1x | Google Colab (miễn phí) |
| Colab V100 | 16GB | 2x | Colab Pro ($10/tháng) |
| Colab A100 | 40GB | 4x | Colab Pro+ ($50/tháng) |
| RTX 3060 | 12GB | 0.8x | Máy bạn (local) |
| RTX 4090 | 24GB | 3x | Máy bạn (local) |

---

## 🐛 Troubleshooting

### Lỗi: "No module named 'transformers'"
```python
!pip install transformers torch datasets
```

### Lỗi: "CUDA out of memory"
Sửa `advanced_config.yaml`:
```yaml
training:
  batch_size: 8  # Giảm từ 16
  gradient_accumulation_steps: 4  # Tăng từ 2
```

### Colab bị disconnect
Thêm cell này để giữ kết nối:
```python
import time
while True:
    time.sleep(60)  # Ping mỗi phút
```

---

## 📝 Template Notebook cho Colab

Tạo notebook mới `colab_train.ipynb`:

```python
# === CELL 1: Setup ===
!pip install -q transformers torch datasets accelerate tensorboard

# === CELL 2: Upload project (nếu chưa có) ===
from google.colab import files
uploaded = files.upload()
!unzip -q transformer-project.zip
%cd transformer-project

# === CELL 3: Check GPU ===
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# === CELL 4: Train ===
!python main.py train --config configs/advanced_config.yaml

# === CELL 5: Evaluate ===
!python main.py predict --checkpoint outputs/advanced_training/best_model
```

---

## 💡 Tips

1. **Lưu checkpoint về Drive** để không mất khi disconnect:
```python
!cp -r outputs/ /content/drive/MyDrive/checkpoints/
```

2. **Monitor với TensorBoard**:
```python
%load_ext tensorboard
%tensorboard --logdir logs/tensorboard
```

3. **Download kết quả về máy**:
```python
from google.colab import files
!zip -r results.zip outputs/
files.download('results.zip')
```

