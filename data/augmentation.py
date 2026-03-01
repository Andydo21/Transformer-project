"""
Data augmentation utilities for Vietnamese text
Các phương pháp tăng cường dữ liệu cho tiếng Việt
"""
import random
import re
from typing import List


class VietnameseTextAugmenter:
    """Text augmentation cho tiếng Việt"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Từ đồng nghĩa phổ biến trong hợp đồng
        self.synonyms = {
            'hợp đồng': ['thỏa thuận', 'văn bản', 'giao kết'],
            'bên': ['phía', 'đơn vị'],
            'ký kết': ['ký', 'giao kết', 'thỏa thuận'],
            'thỏa thuận': ['đồng ý', 'chấp nhận', 'cam kết'],
            'giá trị': ['số tiền', 'trị giá', 'mức giá'],
            'thời hạn': ['thời gian', 'khoảng thời gian', 'kỳ hạn'],
            'điều khoản': ['quy định', 'nội dung', 'điều'],
            'thanh toán': ['chi trả', 'trả tiền', 'toán thanh'],
            'giao hàng': ['chuyển giao', 'bàn giao', 'vận chuyển'],
        }
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Thay thế n từ bằng từ đồng nghĩa
        
        Args:
            text: Text gốc
            n: Số từ cần thay thế
        
        Returns:
            Text đã augment
        """
        words = text.split()
        for _ in range(n):
            for word, synonyms in self.synonyms.items():
                if word in text.lower():
                    synonym = random.choice(synonyms)
                    # Replace case-insensitive
                    text = re.sub(
                        r'\b' + word + r'\b',
                        synonym,
                        text,
                        flags=re.IGNORECASE,
                        count=1
                    )
                    break
        return text
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Xóa ngẫu nhiên các từ với xác suất p
        
        Args:
            text: Text gốc
            p: Xác suất xóa mỗi từ
        
        Returns:
            Text đã augment
        """
        words = text.split()
        if len(words) < 3:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # Đảm bảo ít nhất giữ lại 1 từ
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Hoán đổi ngẫu nhiên vị trí n cặp từ
        
        Args:
            text: Text gốc
            n: Số lần hoán đổi
        
        Returns:
            Text đã augment
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Chèn ngẫu nhiên n từ đồng nghĩa
        
        Args:
            text: Text gốc
            n: Số từ cần chèn
        
        Returns:
            Text đã augment
        """
        words = text.split()
        for _ in range(n):
            # Random chọn một từ có synonym
            available_words = [w for w in words if w.lower() in self.synonyms]
            if not available_words:
                continue
            
            word = random.choice(available_words)
            if word.lower() in self.synonyms:
                synonym = random.choice(self.synonyms[word.lower()])
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, synonym)
        
        return ' '.join(words)
    
    def augment(
        self,
        text: str,
        methods: List[str] = ['synonym', 'deletion', 'swap'],
        num_aug: int = 1
    ) -> List[str]:
        """
        Tạo nhiều phiên bản augmented text
        
        Args:
            text: Text gốc
            methods: Danh sách phương pháp augmentation
            num_aug: Số phiên bản augmented cần tạo
        
        Returns:
            List các text đã augment
        """
        augmented_texts = []
        
        for _ in range(num_aug):
            aug_text = text
            for method in methods:
                if method == 'synonym':
                    aug_text = self.synonym_replacement(aug_text, n=2)
                elif method == 'deletion':
                    aug_text = self.random_deletion(aug_text, p=0.1)
                elif method == 'swap':
                    aug_text = self.random_swap(aug_text, n=2)
                elif method == 'insertion':
                    aug_text = self.random_insertion(aug_text, n=1)
            
            augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def back_translation_placeholder(self, text: str) -> str:
        """
        Placeholder cho back translation (cần API dịch)
        Back translation: VI -> EN -> VI
        """
        # TODO: Implement với Google Translate API hoặc similar
        return text


def augment_dataset(data: List[dict], num_aug_per_sample: int = 2) -> List[dict]:
    """
    Augment toàn bộ dataset
    
    Args:
        data: List các dict có 'text' và 'label'
        num_aug_per_sample: Số augmented samples cho mỗi sample gốc
    
    Returns:
        Augmented dataset
    """
    augmenter = VietnameseTextAugmenter()
    augmented_data = []
    
    for item in data:
        # Giữ nguyên sample gốc
        augmented_data.append(item)
        
        # Tạo augmented samples
        aug_texts = augmenter.augment(
            item['text'],
            methods=['synonym', 'deletion', 'swap'],
            num_aug=num_aug_per_sample
        )
        
        for aug_text in aug_texts:
            augmented_data.append({
                'text': aug_text,
                'label': item['label']
            })
    
    return augmented_data


if __name__ == '__main__':
    # Test augmentation
    augmenter = VietnameseTextAugmenter()
    
    sample_text = "Hợp đồng mua bán hàng hóa được ký kết giữa hai bên với giá trị 500 triệu đồng"
    
    print("Original:")
    print(sample_text)
    print("\nAugmented versions:")
    
    aug_texts = augmenter.augment(sample_text, num_aug=3)
    for i, aug_text in enumerate(aug_texts, 1):
        print(f"\n{i}. {aug_text}")
