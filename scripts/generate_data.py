"""
Generate sample contract data for testing
Tạo dữ liệu hợp đồng mẫu để test
"""
import json
import random
from typing import List, Dict


def generate_contract_samples(num_samples: int = 50) -> List[Dict]:
    """
    Generate synthetic contract samples
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        List of contract samples
    """
    
    # Templates cho các loại hợp đồng
    templates = {
        0: [  # Mua bán
            "HỢP ĐỒNG MUA BÁN HÀNG HÓA số {num}/2026. Bên A là {company_a}, bên B là {company_b}. Đối tượng: {product}. Giá trị: {price} VNĐ. Thời hạn giao hàng: {days} ngày.",
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nHỢP ĐỒNG MUA BÁN\nSố: {num}/2026/HĐMB\nBên mua: {company_a}\nBên bán: {company_b}\nHàng hóa: {product}\nSố lượng: {quantity} đơn vị\nĐơn giá: {unit_price} VNĐ\nTổng giá trị: {price} VNĐ",
            "Hợp đồng mua bán {product} được ký kết ngày {date}. Bên A ({company_a}) cam kết thanh toán {price} VNĐ cho Bên B ({company_b}) trong vòng {days} ngày.",
        ],
        1: [  # Thuê nhà
            "HỢP ĐỒNG THUÊ NHÀ số {num}/2026. Bên cho thuê: {person_a}. Bên thuê: {person_b}. Địa chỉ: {address}. Diện tích: {area} m2. Giá thuê: {rent} VNĐ/tháng. Thời hạn: {months} tháng.",
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nHỢP ĐỒNG THUÊ NHÀ Ở\nSố: {num}/2026\nBên cho thuê: {person_a}, CMND: {id_a}\nBên thuê: {person_b}, CMND: {id_b}\nĐịa chỉ thuê: {address}\nDiện tích: {area}m2\nGiá thuê: {rent} VNĐ/tháng\nĐặt cọc: {deposit} VNĐ",
            "Hợp đồng thuê căn hộ tại {address} giữa {person_a} và {person_b}. Giá {rent} VNĐ/tháng, diện tích {area}m2, thời hạn {months} tháng kể từ {date}.",
        ],
        2: [  # Dịch vụ
            "HỢP ĐỒNG CUNG CẤP DỊCH VỤ số {num}/2026. Bên A: {company_a}. Bên B: {company_b}. Dịch vụ: {service}. Giá trị: {price} VNĐ. Thời hạn: {months} tháng.",
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nHỢP ĐỒNG DỊCH VỤ\nSố: {num}/2026/HĐDV\nKhách hàng: {company_a}\nNhà cung cấp: {company_b}\nLoại dịch vụ: {service}\nGiá trị hợp đồng: {price} VNĐ/năm\nThời gian thực hiện: {months} tháng",
            "Hợp đồng cung cấp {service} giữa {company_a} và {company_b}. Giá trị hợp đồng {price} VNĐ, thanh toán theo {payment_term}. Thời hạn {months} tháng.",
        ]
    }
    
    # Sample data pools
    companies = [
        "Công ty TNHH Thương mại ABC", "Công ty Cổ phần XYZ",
        "Công ty TNHH Công nghệ DEF", "Công ty Cổ phần Sản xuất GHI",
        "Công ty TNHH Dịch vụ JKL", "Công ty Cổ phần Đầu tư MNO",
        "Công ty TNHH Xây dựng PQR", "Công ty Cổ phần Thương mại STU"
    ]
    
    persons = [
        "Ông Nguyễn Văn A", "Bà Trần Thị B", "Ông Lê Văn C",
        "Bà Phạm Thị D", "Ông Hoàng Văn E", "Bà Vũ Thị F",
        "Ông Đặng Văn G", "Bà Bùi Thị H"
    ]
    
    addresses = [
        "123 Đường Láng, Đống Đa, Hà Nội",
        "456 Giải Phóng, Hai Bà Trưng, Hà Nội",
        "789 Nguyễn Trãi, Thanh Xuân, Hà Nội",
        "321 Hoàng Quốc Việt, Cầu Giấy, Hà Nội",
        "555 Láng Hạ, Ba Đình, Hà Nội",
        "777 Trần Duy Hưng, Cầu Giấy, Hà Nội"
    ]
    
    products = [
        "máy tính xách tay", "điện thoại thông minh", "thiết bị điện tử",
        "hàng tiêu dùng", "vật liệu xây dựng", "thiết bị văn phòng",
        "sản phẩm công nghệ", "phụ tùng ô tô"
    ]
    
    services = [
        "bảo trì hệ thống IT", "tư vấn pháp lý", "vệ sinh văn phòng",
        "bảo vệ an ninh", "kiểm toán", "quảng cáo truyền thông",
        "đào tạo nhân sự", "thiết kế website"
    ]
    
    payment_terms = [
        "tháng", "quý", "năm", "giai đoạn", "lần"
    ]
    
    dates = [
        "01/01/2026", "15/02/2026", "01/03/2026", "20/04/2026",
        "10/05/2026", "01/06/2026", "15/07/2026", "01/08/2026"
    ]
    
    # Generate samples
    samples = []
    
    for i in range(num_samples):
        # Random label
        label = random.randint(0, 2)
        
        # Random template
        template = random.choice(templates[label])
        
        # Fill template
        text = template.format(
            num=str(i + 1).zfill(3),
            company_a=random.choice(companies),
            company_b=random.choice(companies),
            person_a=random.choice(persons),
            person_b=random.choice(persons),
            id_a=f"0{random.randint(10000000, 99999999)}",
            id_b=f"0{random.randint(10000000, 99999999)}",
            address=random.choice(addresses),
            product=random.choice(products),
            service=random.choice(services),
            price=f"{random.randint(10, 500)}0,000,000",
            rent=f"{random.randint(5, 50)},000,000",
            deposit=f"{random.randint(10, 100)},000,000",
            area=random.randint(30, 200),
            quantity=random.randint(100, 10000),
            unit_price=f"{random.randint(100, 5000)},000",
            days=random.randint(30, 90),
            months=random.randint(6, 36),
            payment_term=random.choice(payment_terms),
            date=random.choice(dates)
        )
        
        samples.append({
            "text": text,
            "label": label
        })
    
    return samples


def save_samples(samples: List[Dict], output_file: str):
    """Save samples to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(samples)} samples to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample contract data')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/generated_samples.json', help='Output file')
    parser.add_argument('--split', action='store_true', help='Split into train/val/test')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples} samples...")
    samples = generate_contract_samples(args.num_samples)
    
    if args.split:
        # Split into train/val/test
        random.shuffle(samples)
        
        train_size = int(0.7 * len(samples))
        val_size = int(0.15 * len(samples))
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        save_samples(train_samples, 'data/train_samples.json')
        save_samples(val_samples, 'data/val_samples.json')
        save_samples(test_samples, 'data/test_samples.json')
        
        print(f"\nSplit:")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples")
        print(f"  Test:  {len(test_samples)} samples")
    else:
        save_samples(samples, args.output)
    
    print("\n✅ Done!")
