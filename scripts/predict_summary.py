"""
Inference script for ViT5 Contract Summarization

Generate summaries for contract texts

Usage:
    # Single text
    python scripts/predict_summary.py \
        --model outputs/vit5_summarizer/final_model \
        --text "Contract text here..."
    
    # From file
    python scripts/predict_summary.py \
        --model outputs/vit5_summarizer/final_model \
        --input-file data/test.txt \
        --output-file predictions.txt
"""

import sys
import argparse
import json
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.vit5_summarizer import ViT5ContractSummarizer


def predict_single(summarizer, text, args):
    """Predict summary for single text"""
    print("\n" + "=" * 60)
    print("INPUT TEXT:")
    print("=" * 60)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print(f"\nLength: {len(text)} characters")
    
    print("\nGenerating summary...")
    summary = summarizer.generate_summary(
        text,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(summary)
    print(f"\nLength: {len(summary)} characters")
    print(f"Compression ratio: {len(summary) / len(text):.1%}")
    
    return summary


def predict_batch(summarizer, input_file, output_file, args):
    """Predict summaries for multiple texts"""
    print(f"\nReading from {input_file}")
    
    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                texts = [item['text'] if isinstance(item, dict) else item for item in data]
            else:
                texts = [data['text']]
        else:
            texts = [f.read()]
    
    print(f"Processing {len(texts)} text(s)...")
    
    # Generate summaries
    summaries = []
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Generating summary...")
        summary = summarizer.generate_summary(
            text,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty
        )
        summaries.append(summary)
        
        print(f"  Input length: {len(text)} chars")
        print(f"  Summary length: {len(summary)} chars")
        print(f"  Compression: {len(summary)/len(text):.1%}")
    
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_file.endswith('.json'):
            output_data = [
                {
                    "text": text,
                    "summary": summary,
                    "length_input": len(text),
                    "length_summary": len(summary),
                    "compression_ratio": len(summary) / len(text)
                }
                for text, summary in zip(texts, summaries)
            ]
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:
            for i, (text, summary) in enumerate(zip(texts, summaries)):
                f.write(f"=== Text {i+1} ===\n")
                f.write(f"{text}\n\n")
                f.write(f"=== Summary {i+1} ===\n")
                f.write(f"{summary}\n\n")
                f.write("-" * 60 + "\n\n")
    
    print(f"\n✓ Summaries saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ViT5 Summarization Inference")
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    
    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str,
                      help="Input text to summarize")
    group.add_argument("--input-file", type=str,
                      help="Input file (.txt or .json)")
    
    # Output
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file (for batch processing)")
    
    # Generation
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum summary length")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum summary length")
    parser.add_argument("--num-beams", type=int, default=4,
                       help="Beam search width")
    parser.add_argument("--length-penalty", type=float, default=2.0,
                       help="Length penalty (>1 = longer summaries)")
    
    args = parser.parse_args()
    
    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    print(f"Model path: {args.model}")
    
    summarizer = ViT5ContractSummarizer(
        model_name=args.model,
        max_input_length=1024,
        max_output_length=args.max_length,
        min_output_length=args.min_length
    )
    
    # Generate summaries
    if args.text:
        # Single text
        predict_single(summarizer, args.text, args)
    
    else:
        # Batch from file
        if not args.output_file:
            # Auto-generate output filename
            input_path = Path(args.input_file)
            args.output_file = str(input_path.parent / f"{input_path.stem}_summaries.txt")
        
        predict_batch(summarizer, args.input_file, args.output_file, args)


if __name__ == "__main__":
    main()
