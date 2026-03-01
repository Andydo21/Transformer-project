"""
Generate sample data for contract summarization training

Creates synthetic Vietnamese contract data with summaries
"""

import json
import random
from pathlib import Path
import argparse


# Contract templates with summaries
TEMPLATES = [
    {
        "text": """HỢP ĐỒNG CUNG CẤP DỊCH VỤ

Hôm nay, ngày {day}/{month}/2026, tại {city}, chúng tôi gồm:

BÊN A (BÊN CUNG CẤP DỊCH VỤ):
Công ty: {company_a}
Địa chỉ: {address_a}
Mã số thuế: {tax_a}
Người đại diện: {rep_a}

BÊN B (BÊN THUÊ DỊCH VỤ):
Công ty: {company_b}
Địa chỉ: {address_b}
Mã số thuế: {tax_b}
Người đại diện: {rep_b}

Hai bên thỏa thuận ký kết hợp đồng cung cấp dịch vụ {service} với các điều khoản sau:

ĐIỀU 1: NỘI DUNG DỊCH VỤ
Bên A cam kết cung cấp dịch vụ {service} cho Bên B theo đúng tiêu chuẩn chất lượng đã thỏa thuận.
Phạm vi công việc bao gồm {scope}.
Bên A sẽ triển khai dịch vụ tại {location}.

ĐIỀU 2: GIÁ TRỊ HỢP ĐỒNG VÀ THANH TOÁN
Tổng giá trị hợp đồng: {value} đồng (Bằng chữ: {value_text}).
Phương thức thanh toán: {payment_method}.
Bên B thanh toán {payment_1}% khi ký hợp đồng, {payment_2}% sau khi hoàn thành.

ĐIỀU 3: THỜI HẠN THỰC HIỆN
Thời gian bắt đầu: {start_date}
Thời gian kết thúc: {end_date}
Thời hạn hợp đồng là {duration} tháng.

ĐIỀU 4: TRÁCH NHIỆM CỦA CÁC BÊN
Bên A có trách nhiệm thực hiện dịch vụ đúng chất lượng và tiến độ cam kết.
Bên A được quyền yêu cầu thanh toán đầy đủ theo thỏa thuận.
Bên B có trách nhiệm thanh toán đúng hạn và cung cấp thông tin cần thiết.
Bên B có quyền yêu cầu Bên A thực hiện đúng cam kết.

ĐIỀU 5: ĐIỀU KHOẢN CHUNG
Hợp đồng có hiệu lực kể từ ngày ký.
Mọi tranh chấp phát sinh sẽ được giải quyết thông qua thương lượng.
Nếu không thỏa thuận được, hai bên sẽ đưa ra tòa án có thẩm quyền giải quyết.
Hợp đồng được lập thành 02 bản, mỗi bên giữ 01 bản có giá trị pháp lý như nhau.""",
        
        "summary": "Hợp đồng cung cấp dịch vụ {service} giữa {company_a} và {company_b} được ký kết tại {city} ngày {day}/{month}/2026. Giá trị hợp đồng {value} đồng, thời hạn {duration} tháng, thanh toán {payment_1}% trước và {payment_2}% sau khi hoàn thành. Bên A cam kết thực hiện đúng chất lượng và tiến độ, Bên B có trách nhiệm thanh toán đúng hạn."
    },
    {
        "text": """HỢP ĐỒNG MUA BÁN HÀNG HÓA

Căn cứ Bộ luật Dân sự năm 2015 và các quy định pháp luật có liên quan, hôm nay ngày {day}/{month}/2026, tại {city}, chúng tôi gồm:

BÊN BÁN (BÊN A):
Công ty: {company_a}
Địa chỉ: {address_a}
Mã số thuế: {tax_a}
Đại diện: {rep_a}

BÊN MUA (BÊN B):
Công ty: {company_b}
Địa chỉ: {address_b}
Mã số thuế: {tax_b}
Đại diện: {rep_b}

Hai bên thống nhất ký kết hợp đồng mua bán {product} với nội dung sau:

ĐIỀU 1: ĐỐI TƯỢNG HỢP ĐỒNG
Bên A đồng ý bán và Bên B đồng ý mua {product}.
Số lượng: {quantity} {unit}
Chất lượng: {quality}
Xuất xứ: {origin}

ĐIỀU 2: GIÁ TRỊ VÀ THANH TOÁN
Đơn giá: {unit_price} đồng/{unit}
Tổng giá trị: {value} đồng ({value_text})
Phương thức thanh toán: {payment_method}
Thời hạn thanh toán: {payment_term}

ĐIỀU 3: BÀN GIAO VÀ VẬN CHUYỂN
Địa điểm giao hàng: {delivery_location}
Thời gian giao hàng: {delivery_time}
Chi phí vận chuyển: {shipping_cost}
Bên A chịu trách nhiệm đóng gói và bảo quản hàng hóa trong quá trình vận chuyển.

ĐIỀU 4: BẢO HÀNH VÀ BẢO HIỂM
Thời gian bảo hành: {warranty} tháng
Điều kiện bảo hành: Theo quy định của nhà sản xuất
Bảo hiểm hàng hóa: {insurance}

ĐIỀU 5: CAM KẾT
Bên A cam kết hàng hóa đúng chất lượng, số lượng và thời gian giao hàng.
Bên B cam kết thanh toán đầy đủ và đúng hạn theo thỏa thuận.""",
        
        "summary": "Hợp đồng mua bán {product} giữa {company_a} (bên bán) và {company_b} (bên mua) tại {city} ngày {day}/{month}/2026. Số lượng {quantity} {unit}, tổng giá trị {value} đồng. Giao hàng tại {delivery_location} trong {delivery_time}, bảo hành {warranty} tháng. Thanh toán {payment_method} trong {payment_term}."
    }
]

# Data for generation
COMPANIES = [
    "Công ty TNHH ABC", "Công ty CP XYZ", "Công ty TNHH Đầu Tư DEF",
    "Công ty CP Công Nghệ GHI", "Công ty TNHH Thương Mại JKL",
    "Công ty CP Sản Xuất MNO", "Công ty TNHH Dịch Vụ PQR"
]

SERVICES = [
    "tư vấn phát triển phần mềm", "quản lý và bảo trì hệ thống",
    "thiết kế và thi công nội thất", "cung cấp nhân sự IT",
    "marketing và quảng cáo", "kế toán và kiểm toán",
    "đào tạo nhân sự", "tư vấn pháp lý doanh nghiệp"
]

PRODUCTS = [
    "máy tính xách tay", "thiết bị mạng", "điện thoại thông minh",
    "máy in văn phòng", "thiết bị chiếu", "phần mềm quản lý",
    "vật liệu xây dựng", "nguyên liệu sản xuất"
]

CITIES = ["TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]

ADDRESSES = [
    "123 Đường Nguyễn Văn Linh, Quận 7",
    "456 Đường Lê Văn Việt, Quận 9",
    "789 Đường Trần Hưng Đạo, Quận 1",
    "321 Đường Võ Văn Kiệt, Quận 5"
]


def generate_sample():
    """Generate one contract sample"""
    template = random.choice(TEMPLATES)
    
    # Generate values
    payment_1 = random.choice([30, 40, 50])
    payment_2 = 100 - payment_1
    
    values = {
        "day": f"{random.randint(1, 28):02d}",
        "month": f"{random.randint(1, 12):02d}",
        "city": random.choice(CITIES),
        "company_a": random.choice(COMPANIES),
        "company_b": random.choice([c for c in COMPANIES]),
        "address_a": random.choice(ADDRESSES),
        "address_b": random.choice(ADDRESSES),
        "tax_a": f"{random.randint(1000000000, 9999999999)}",
        "tax_b": f"{random.randint(1000000000, 9999999999)}",
        "rep_a": random.choice(["Ông Nguyễn Văn A", "Bà Trần Thị B", "Ông Lê Văn C"]),
        "rep_b": random.choice(["Ông Phạm Văn D", "Bà Hoàng Thị E", "Ông Đỗ Văn F"]),
        "service": random.choice(SERVICES),
        "product": random.choice(PRODUCTS),
        "scope": "phân tích, thiết kế, lập trình và triển khai",
        "location": random.choice(["trụ sở Bên B", "theo yêu cầu của Bên B"]),
        "value": f"{random.randint(50, 999)}.000.000",
        "value_text": random.choice(["năm mươi", "một trăm", "hai trăm", "ba trăm", "năm trăm"]) + " triệu đồng",
        "payment_method": random.choice(["Chuyển khoản ngân hàng", "Tiền mặt", "Séc"]),
        "payment_1": payment_1,
        "payment_2": payment_2,
        "start_date": "01/03/2026",
        "end_date": "31/12/2026",
        "duration": random.randint(6, 24),
        "quantity": random.randint(10, 100),
        "unit": random.choice(["chiếc", "bộ", "cái", "tấn", "kg"]),
        "quality": random.choice(["Hàng chính hãng", "Đạt tiêu chuẩn quốc gia", "Loại A"]),
        "origin": random.choice(["Việt Nam", "Nhật Bản", "Hàn Quốc", "Châu Âu"]),
        "unit_price": f"{random.randint(1, 99)}.000",
        "delivery_location": random.choice(["Kho Bên B", "Theo địa chỉ Bên B"]),
        "delivery_time": random.choice(["30 ngày", "45 ngày", "60 ngày"]),
        "shipping_cost": random.choice(["Bên A chi trả", "Bên B chi trả", "Thỏa thuận riêng"]),
        "warranty": random.choice([12, 18, 24, 36]),
        "insurance": random.choice(["Bên A mua bảo hiểm", "Không bảo hiểm", "Theo thỏa thuận"]),
        "payment_term": random.choice(["7 ngày", "15 ngày", "30 ngày"])
    }
    
    # Fill template
    text = template["text"].format(**values)
    summary = template["summary"].format(**values)
    
    return {
        "text": text.strip(),
        "summary": summary.strip(),
        "metadata": {
            "length_input": len(text),
            "length_summary": len(summary),
            "compression_ratio": len(summary) / len(text)
        }
    }


def generate_dataset(num_samples, output_dir, split=True):
    """Generate full dataset"""
    print(f"Generating {num_samples} samples...")
    
    samples = []
    for i in range(num_samples):
        sample = generate_sample()
        samples.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if split:
        # Split into train/val/test
        random.shuffle(samples)
        
        train_size = int(num_samples * 0.8)
        val_size = int(num_samples * 0.1)
        
        train_data = samples[:train_size]
        val_data = samples[train_size:train_size + val_size]
        test_data = samples[train_size + val_size:]
        
        # Save splits
        with open(output_path / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Dataset saved to {output_dir}/")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Val: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
    
    else:
        # Save all in one file
        with open(output_path / "all.json", 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Dataset saved to {output_dir}/all.json")
        print(f"  Total: {len(samples)} samples")
    
    # Print example
    print("\n" + "=" * 60)
    print("EXAMPLE SAMPLE:")
    print("=" * 60)
    example = samples[0]
    print(f"\nInput text ({example['metadata']['length_input']} chars):")
    print(example['text'][:300] + "...")
    print(f"\nSummary ({example['metadata']['length_summary']} chars):")
    print(example['summary'])
    print(f"\nCompression ratio: {example['metadata']['compression_ratio']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Generate Summarization Data")
    
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of samples to generate")
    
    parser.add_argument("--output-dir", type=str, default="data/summarization",
                       help="Output directory")
    
    parser.add_argument("--split", action="store_true",
                       help="Split into train/val/test")
    
    args = parser.parse_args()
    
    generate_dataset(args.num_samples, args.output_dir, args.split)


if __name__ == "__main__":
    main()
