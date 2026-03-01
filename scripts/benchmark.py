"""
Benchmark script to measure model performance
Script đo performance của model
"""
import os
import sys
import time
import torch
import numpy as np
from typing import List, Dict
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import ContractPredictor
from utils.logger import setup_logger


def benchmark_latency(
    predictor: ContractPredictor,
    texts: List[str],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict:
    """
    Benchmark inference latency
    
    Args:
        predictor: ContractPredictor instance
        texts: List of sample texts
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with benchmark results
    """
    logger = setup_logger('benchmark')
    
    # Warmup
    logger.info(f"Warmup: {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = predictor.predict(texts[0])
    
    # Benchmark
    logger.info(f"Benchmarking: {num_runs} runs...")
    latencies = []
    
    for i in range(num_runs):
        text = texts[i % len(texts)]
        
        start_time = time.time()
        _ = predictor.predict(text)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{num_runs}")
    
    # Calculate statistics
    results = {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p90_latency_ms': np.percentile(latencies, 90),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_qps': 1000 / np.mean(latencies)  # queries per second
    }
    
    return results


def benchmark_memory():
    """Benchmark memory usage"""
    if torch.cuda.is_available():
        return {
            'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
        }
    return {'note': 'GPU not available'}


def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of runs')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup runs')
    
    args = parser.parse_args()
    
    logger = setup_logger('benchmark')
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    # Load predictor
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = ContractPredictor(args.checkpoint)
    
    # Sample texts
    sample_texts = [
        "HỢP ĐỒNG MUA BÁN hàng hóa số 001/2026",
        "HỢP ĐỒNG THUÊ nhà ở khu vực Hà Nội",
        "HỢP ĐỒNG CUNG CẤP DỊCH VỤ bảo trì hệ thống",
        "Hợp đồng giao kết giữa hai bên A và B",
        "Thỏa thuận về giá trị và thời hạn thực hiện"
    ]
    
    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Benchmark latency
    latency_results = benchmark_latency(
        predictor,
        sample_texts,
        num_runs=args.num_runs,
        warmup_runs=args.warmup
    )
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("LATENCY RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean:       {latency_results['mean_latency_ms']:.2f} ms")
    logger.info(f"Median:     {latency_results['median_latency_ms']:.2f} ms")
    logger.info(f"Std Dev:    {latency_results['std_latency_ms']:.2f} ms")
    logger.info(f"Min:        {latency_results['min_latency_ms']:.2f} ms")
    logger.info(f"Max:        {latency_results['max_latency_ms']:.2f} ms")
    logger.info(f"P50:        {latency_results['p50_latency_ms']:.2f} ms")
    logger.info(f"P90:        {latency_results['p90_latency_ms']:.2f} ms")
    logger.info(f"P95:        {latency_results['p95_latency_ms']:.2f} ms")
    logger.info(f"P99:        {latency_results['p99_latency_ms']:.2f} ms")
    logger.info(f"Throughput: {latency_results['throughput_qps']:.2f} QPS")
    
    # Memory results
    memory_results = benchmark_memory()
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY USAGE")
    logger.info("=" * 60)
    for key, value in memory_results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f} MB")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETED")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
