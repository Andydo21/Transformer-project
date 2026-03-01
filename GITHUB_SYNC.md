# 🔄 Sync Code giữa VS Code Local và Colab qua GitHub

## 📋 Setup lần đầu (5 phút)

### 1. Tạo repo trên GitHub
1. Vào https://github.com/new
2. Tạo repo tên: `transformer-project` (hoặc tên khác)
3. **Chọn:** Private (nếu muốn giữ riêng tư)
4. **KHÔNG** tích "Add README"
5. Click "Create repository"

### 2. Push code lên GitHub từ VS Code

Mở Terminal trong VS Code (`Ctrl+``) và chạy:

```bash
# Đảm bảo đang ở thư mục transformer-project/
cd d:\Django_project\contract_dl\transformer-project

# Init git (nếu chưa có)
git init

# Thêm tất cả files
git add .

# Commit
git commit -m "Initial commit"

# Thêm remote (ĐỔI URL thành repo của bạn)
git remote add origin https://github.com/YOUR_USERNAME/transformer-project.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

### 3. Sửa URL trong notebook

Mở [train_notebook.ipynb](train_notebook.ipynb), tìm cell **"Option A: Clone từ GitHub"**, sửa dòng:

```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/transformer-project.git"  # ⚠️ ĐỔI URL NÀY
```

### 4. Chạy trong Colab

1. Kết nối Colab Runtime trong VS Code
2. Chạy cell "Option A"
3. Xong! ✅

---

## 🔄 Workflow hàng ngày (30 giây)

### Khi sửa code trong VS Code local:

```bash
git add .
git commit -m "Updated training config"
git push
```

### Trong Colab:

Chạy lại cell "Option A" → Tự động `git pull` code mới về!

---

## ✅ Ưu điểm so với upload ZIP:

| | GitHub | Upload ZIP |
|---|---|---|
| **Tốc độ sync** | 5-10 giây | 1-2 phút |
| **Tự động** | `git pull` | Phải upload lại |
| **Chỉ cập nhật** | File thay đổi | Toàn bộ |
| **Version control** | ✅ Yes | ❌ No |
| **Backup** | ✅ Yes | ❌ No |

---

## 🔒 Nếu repo là Private (cần access token)

1. Vào https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Chọn scope: `repo` (full control)
4. Copy token (dạng: `ghp_xxxxx`)
5. Trong Colab cell, sửa URL:

```python
GITHUB_REPO = "https://YOUR_TOKEN@github.com/YOUR_USERNAME/transformer-project.git"
```

Hoặc dùng secrets của Colab:

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')  # Set token trong Colab Secrets
GITHUB_REPO = f"https://{token}@github.com/YOUR_USERNAME/transformer-project.git"
```

---

## 🆘 Troubleshooting

### Lỗi: "remote: Repository not found"
→ Repo chưa tồn tại hoặc URL sai. Kiểm tra lại URL.

### Lỗi: "Authentication failed"
→ Repo private cần access token (xem phần trên).

### Lỗi: "fatal: not a git repository"
→ Chưa chạy `git init`. Chạy lại bước 2.

### Tôi sửa code trong Colab, làm sao pull về local?
→ Trong VS Code terminal:
```bash
git pull
```

---

## 💡 Tips

- **Ignore files:** Tạo `.gitignore` để không push file lớn:
  ```
  *.pyc
  __pycache__/
  *.ipynb_checkpoints
  models/*.bin
  models/*.safetensors
  data/
  logs/
  ```

- **Nhiều người cùng làm:** Trước khi `push`, luôn `pull` trước:
  ```bash
  git pull
  git add .
  git commit -m "message"
  git push
  ```

- **Xem thay đổi:**
  ```bash
  git status
  git diff
  ```

---

**🎯 Kết quả:** Bạn code ở VS Code local → `git push` → Colab `git pull` → Chạy training trên GPU T4 miễn phí! 🚀
