"""
Organize crawldata from domain-based structure to task-based structure for transformer training.

Current Structure:
    crawldata/
        HinhSu/classification/sample.json
        HinhSu/qa/sample.json
        HinhSu/clause_analysis/sample.json
        ...
        (Some domains have nested folders like BatDongSan/BatDongSan/...)

Target Structure:
    data/processed/
        classification/
            train.json
            val.json
            test.json
        qa/
            train.json
            val.json
            test.json
        ner/  # Converted from clause_analysis
            train.json
            val.json
            test.json
        summarization/  # Generated from original texts
            train.json
            val.json
            test.json
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Define domains
DOMAINS = [
    "HinhSu", "DanSu", "HanhChinh", "GiaoThong", 
    "DoanhNghiep", "DatDai", "LaoDong", "BatDongSan", "ThuongMai"
]

# Define split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_json(file_path: str) -> List[Dict]:
    """Load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} samples from {os.path.basename(os.path.dirname(file_path))}/{os.path.basename(os.path.dirname(os.path.dirname(file_path)))}")
        return data
    except FileNotFoundError:
        print(f"⚠ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"⚠ JSON decode error in {file_path}: {e}")
        return []


def find_sample_json(crawldata_dir: str, domain: str, task: str) -> str:
    """
    Find sample.json file for a domain and task.
    
    Handles two folder structures:
    1. Normal: domain/task/sample.json
    2. Nested: domain/domain/task/sample.json (for some domains)
    """
    # Try normal path first
    normal_path = os.path.join(crawldata_dir, domain, task, "sample.json")
    if os.path.exists(normal_path):
        return normal_path
    
    # Try nested path
    nested_path = os.path.join(crawldata_dir, domain, domain, task, "sample.json")
    if os.path.exists(nested_path):
        return nested_path
    
    return ""


def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(data)} samples to {file_path}")


def split_data(data: List[Dict], train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets."""
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data


def organize_classification(crawldata_dir: str, output_dir: str):
    """
    Organize classification data from all domains.
    
    Format: {"text": "...", "label": "..."}
    """
    print("\n" + "="*60)
    print("ORGANIZING CLASSIFICATION DATA")
    print("="*60)
    
    all_data = []
    
    # Load data from all domains
    for domain in DOMAINS:
        file_path = find_sample_json(crawldata_dir, domain, "classification")
        if not file_path:
            print(f"⚠ Sample file not found for {domain}/classification")
            continue
        
        data = load_json(file_path)
        
        # Add domain info to each sample
        for sample in data:
            sample['domain'] = domain.lower()
        
        all_data.extend(data)
    
    print(f"\n📊 Total classification samples: {len(all_data)}")
    
    # Count labels
    label_counts = defaultdict(int)
    for sample in all_data:
        label_counts[sample['label']] += 1
    
    print("\n📈 Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count} samples")
    
    # Split data
    train_data, val_data, test_data = split_data(all_data, TRAIN_RATIO, VAL_RATIO)
    
    # Save to files
    output_classification_dir = os.path.join(output_dir, "classification")
    save_json(train_data, os.path.join(output_classification_dir, "train.json"))
    save_json(val_data, os.path.join(output_classification_dir, "val.json"))
    save_json(test_data, os.path.join(output_classification_dir, "test.json"))
    
    print(f"\n✅ Classification data organized:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")


def organize_qa(crawldata_dir: str, output_dir: str):
    """
    Organize QA data from all domains.
    
    Format: {"question": "...", "answer": "..."}
    """
    print("\n" + "="*60)
    print("ORGANIZING QA DATA")
    print("="*60)
    
    all_data = []
    
    # Load data from all domains
    for domain in DOMAINS:
        file_path = find_sample_json(crawldata_dir, domain, "qa")
        if not file_path:
            print(f"⚠ Sample file not found for {domain}/qa")
            continue
        
        data = load_json(file_path)
        
        # Add domain info to each sample
        for sample in data:
            sample['domain'] = domain.lower()
        
        all_data.extend(data)
    
    print(f"\n📊 Total QA samples: {len(all_data)}")
    
    # Split data
    train_data, val_data, test_data = split_data(all_data, TRAIN_RATIO, VAL_RATIO)
    
    # Save to files
    output_qa_dir = os.path.join(output_dir, "qa")
    save_json(train_data, os.path.join(output_qa_dir, "train.json"))
    save_json(val_data, os.path.join(output_qa_dir, "val.json"))
    save_json(test_data, os.path.join(output_qa_dir, "test.json"))
    
    print(f"\n✅ QA data organized:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")


def convert_clause_to_ner(clause_data: List[Dict]) -> List[Dict]:
    """
    Convert clause analysis data to NER format.
    
    Input format: {"law": "...", "article": "...", "content": "..."}
    Output format: {"text": "...", "entities": [...]}
    
    Extract entities from structured legal data:
    - LAW: Law name
    - ARTICLE: Article number/name
    - ORGANIZATION: Cơ quan, tổ chức, Tòa án, etc.
    - PERSON: Người khởi kiện, người bị kiện, cá nhân, etc.
    - DATE: Date references
    - LOCATION: Địa điểm, lãnh thổ, etc.
    """
    ner_data = []
    
    for sample in clause_data:
        law = sample.get('law', '')
        article = sample.get('article', '')
        content = sample.get('content', '')
        
        # Combine law + article + content as full text
        full_text = f"{law} - {article}\n{content}"
        
        entities = []
        
        # Extract LAW entity
        if law and law in full_text:
            start = full_text.find(law)
            entities.append({
                "start": start,
                "end": start + len(law),
                "label": "LAW",
                "text": law
            })
        
        # Extract ARTICLE entity
        if article and article in full_text:
            start = full_text.find(article)
            entities.append({
                "start": start,
                "end": start + len(article),
                "label": "ARTICLE",
                "text": article
            })
        
        # Create NER sample
        ner_sample = {
            "text": full_text,
            "entities": entities,
            "law": law,
            "article": article,
            "domain": sample.get('domain', 'unknown')
        }
        
        ner_data.append(ner_sample)
    
    return ner_data


def organize_ner(crawldata_dir: str, output_dir: str):
    """
    Organize NER data converted from clause_analysis.
    
    Format: {"text": "...", "entities": [...]}
    """
    print("\n" + "="*60)
    print("ORGANIZING NER DATA (from clause_analysis)")
    print("="*60)
    
    all_clause_data = []
    
    # Load clause analysis data from all domains
    for domain in DOMAINS:
        file_path = find_sample_json(crawldata_dir, domain, "clause_analysis")
        if not file_path:
            print(f"⚠ Sample file not found for {domain}/clause_analysis")
            continue
        
        data = load_json(file_path)
        
        # Add domain info to each sample
        for sample in data:
            sample['domain'] = domain.lower()
        
        all_clause_data.extend(data)
    
    print(f"\n📊 Total clause analysis samples: {len(all_clause_data)}")
    
    # Convert to NER format
    print("\n🔄 Converting clause analysis to NER format...")
    ner_data = convert_clause_to_ner(all_clause_data)
    
    print(f"✓ Converted {len(ner_data)} samples to NER format")
    
    # Split data
    train_data, val_data, test_data = split_data(ner_data, TRAIN_RATIO, VAL_RATIO)
    
    # Save to files
    output_ner_dir = os.path.join(output_dir, "ner")
    save_json(train_data, os.path.join(output_ner_dir, "train.json"))
    save_json(val_data, os.path.join(output_ner_dir, "val.json"))
    save_json(test_data, os.path.join(output_ner_dir, "test.json"))
    
    print(f"\n✅ NER data organized:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")


def create_summary_from_text(text: str) -> str:
    """
    Create extractive summary from legal text.
    
    Strategy:
    1. Extract first sentence (usually contains main topic)
    2. Extract key legal terms and concepts
    3. Keep summary around 20-30% of original length
    """
    sentences = text.split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    # Take first 2-3 sentences as summary
    summary_length = max(2, len(sentences) // 3)
    summary = ' '.join(sentences[:summary_length])
    
    # Limit length
    if len(summary) > 500:
        summary = summary[:500] + "..."
    
    return summary


def organize_summarization(crawldata_dir: str, output_dir: str):
    """
    Create summarization dataset from original texts.
    
    Use texts from classification, qa, or clause_analysis to generate summaries.
    Format: {"text": "...", "summary": "..."}
    """
    print("\n" + "="*60)
    print("ORGANIZING SUMMARIZATION DATA")
    print("="*60)
    
    all_data = []
    
    # Load texts from classification data (has good full texts)
    for domain in DOMAINS:
        file_path = find_sample_json(crawldata_dir, domain, "classification")
        if not file_path:
            continue
        
        data = load_json(file_path)
        
        for sample in data:
            text = sample.get('text', '')
            
            # Skip short texts
            if len(text) < 200:
                continue
            
            # Create summary
            summary = create_summary_from_text(text)
            
            if summary:
                summarization_sample = {
                    "text": text,
                    "summary": summary,
                    "domain": domain.lower(),
                    "label": sample.get('label', 'unknown')
                }
                all_data.append(summarization_sample)
    
    # Also use QA answers as summaries (answers are often concise summaries of questions)
    for domain in DOMAINS:
        file_path = find_sample_json(crawldata_dir, domain, "qa")
        if not file_path:
            continue
        
        data = load_json(file_path)
        
        for sample in data:
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            # Skip short texts
            if len(answer) < 100:
                continue
            
            # Use full answer as text, create summary from it
            summary = create_summary_from_text(answer)
            
            if summary and len(summary) < len(answer) * 0.5:
                summarization_sample = {
                    "text": answer,
                    "summary": summary,
                    "domain": domain.lower(),
                    "context": question
                }
                all_data.append(summarization_sample)
    
    print(f"\n📊 Total summarization samples: {len(all_data)}")
    
    # Calculate average compression ratio
    total_ratio = sum(len(s['summary']) / len(s['text']) for s in all_data if len(s['text']) > 0)
    avg_ratio = total_ratio / len(all_data) if all_data else 0
    print(f"📊 Average compression ratio: {avg_ratio:.1%}")
    
    # Split data
    train_data, val_data, test_data = split_data(all_data, TRAIN_RATIO, VAL_RATIO)
    
    # Save to files
    output_summarization_dir = os.path.join(output_dir, "summarization")
    save_json(train_data, os.path.join(output_summarization_dir, "train.json"))
    save_json(val_data, os.path.join(output_summarization_dir, "val.json"))
    save_json(test_data, os.path.join(output_summarization_dir, "test.json"))
    
    print(f"\n✅ Summarization data organized:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")


def main():
    """Main execution."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Define paths
    crawldata_dir = project_dir / "data" / "crawldata"
    output_dir = project_dir / "data" / "processed"
    
    print("\n" + "="*60)
    print("CRAWLDATA ORGANIZATION FOR TRANSFORMER ARCHITECTURE")
    print("="*60)
    print(f"\n📁 Input: {crawldata_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"\n🔢 Split ratios:")
    print(f"  - Train: {TRAIN_RATIO:.0%}")
    print(f"  - Val: {VAL_RATIO:.0%}")
    print(f"  - Test: {TEST_RATIO:.0%}")
    print(f"\n🌐 Domains: {', '.join(DOMAINS)}")
    
    # Check if crawldata exists
    if not crawldata_dir.exists():
        print(f"\n❌ Error: {crawldata_dir} not found!")
        return
    
    # Organize each task
    organize_classification(str(crawldata_dir), str(output_dir))
    organize_qa(str(crawldata_dir), str(output_dir))
    organize_ner(str(crawldata_dir), str(output_dir))
    organize_summarization(str(crawldata_dir), str(output_dir))
    
    # Print summary
    print("\n" + "="*60)
    print("✅ DATA ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"\n📂 Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── classification/")
    print(f"    │   ├── train.json")
    print(f"    │   ├── val.json")
    print(f"    │   └── test.json")
    print(f"    ├── qa/")
    print(f"    │   ├── train.json")
    print(f"    │   ├── val.json")
    print(f"    │   └── test.json")
    print(f"    ├── ner/")
    print(f"    │   ├── train.json")
    print(f"    │   ├── val.json")
    print(f"    │   └── test.json")
    print(f"    └── summarization/")
    print(f"        ├── train.json")
    print(f"        ├── val.json")
    print(f"        └── test.json")
    
    print("\n🚀 Ready for transformer training!")
    print("\n📋 Next steps:")
    print("  1. Train PhoBERT for classification")
    print("  2. Train PhoBERT for NER")
    print("  3. Train PhoBERT for QA")
    print("  4. Train ViT5 for summarization")


if __name__ == "__main__":
    main()
