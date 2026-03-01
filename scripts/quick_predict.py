"""
Quick prediction script - Test model nhanh
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import ContractPredictor
from utils.logger import setup_logger


def quick_predict():
    """Quick prediction với sample text"""
    logger = setup_logger('quick_predict')
    
    checkpoint = 'outputs/best_model'
    
    if not os.path.exists(checkpoint):
        logger.error(f"Checkpoint không tồn tại: {checkpoint}")
        logger.info("Chạy training trước: python main.py train")
        return
    
    logger.info("Loading predictor...")
    predictor = ContractPredictor(checkpoint)
    
    # Sample texts
    samples = [
        {
            "text": "HỢP ĐỒNG MUA BÁN\nBên A: Công ty ABC\nBên B: Công ty XYZ\nGiá trị: 100 triệu VNĐ",
            "expected": "Hợp đồng mua bán"
        },
        {
            "text": "HỢP ĐỒNG THUÊ NHÀ\nBên cho thuê: Ông A\nBên thuê: Bà B\nGiá thuê: 10 triệu/tháng",
            "expected": "Hợp đồng thuê"
        },
        {
            "text": "HỢP ĐỒNG DỊCH VỤ\nCung cấp dịch vụ bảo trì\nThời hạn: 12 tháng",
            "expected": "Hợp đồng dịch vụ"
        }
    ]
    
    print("\n" + "="*60)
    print("QUICK PREDICTION TEST")
    print("="*60)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Text: {sample['text'][:100]}...")
        print(f"Expected: {sample['expected']}")
        
        result = predictor.predict(sample['text'], return_probs=True)
        print(f"Predicted: Label {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        if 'label_name' in result:
            print(f"Label Name: {result['label_name']}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    quick_predict()
