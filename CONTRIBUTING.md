# Contributing to PhoBERT Contract Processing

Cảm ơn bạn quan tâm đến việc đóng góp cho dự án! 🎉

## Cách đóng góp

### Báo cáo lỗi (Bug Reports)

Nếu bạn tìm thấy lỗi, vui lòng tạo issue với các thông tin sau:

1. **Mô tả lỗi**: Mô tả chi tiết lỗi
2. **Cách tái hiện**: Các bước để tái hiện lỗi
3. **Kết quả mong đợi**: Bạn mong đợi điều gì xảy ra
4. **Kết quả thực tế**: Điều gì thực sự xảy ra
5. **Môi trường**: OS, Python version, PyTorch version
6. **Screenshots**: Nếu có

### Đề xuất tính năng (Feature Requests)

Để đề xuất tính năng mới:

1. Kiểm tra xem tính năng đã được đề xuất chưa
2. Tạo issue với tag `enhancement`
3. Mô tả chi tiết tính năng và lý do cần thiết
4. Nếu có thể, đề xuất cách implement

### Pull Requests

1. **Fork repository**
   ```bash
   git clone https://github.com/yourusername/transformer-project.git
   ```

2. **Tạo branch mới**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Commit changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```

4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Mở Pull Request**

### Quy tắc code

- Tuân theo PEP 8 style guide
- Viết docstrings cho functions và classes
- Thêm type hints khi có thể
- Viết unit tests cho code mới
- Đảm bảo tests pass trước khi submit

### Quy tắc commit message

- Sử dụng present tense ("Add feature" not "Added feature")
- Giới hạn dòng đầu tiên ở 50 ký tự
- Có thể thêm mô tả chi tiết ở dòng thứ 3+

Ví dụ:
```
Add support for multi-GPU training

- Implement DistributedDataParallel
- Update training script
- Add documentation
```

### Chạy tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=. tests/
```

### Code Review Process

1. Maintainer sẽ review code của bạn
2. Có thể yêu cầu thay đổi
3. Sau khi approved, code sẽ được merge

## Câu hỏi?

Nếu có câu hỏi, tạo issue hoặc liên hệ qua email.

Cảm ơn bạn đã đóng góp! 🙏
