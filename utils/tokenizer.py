"""
Tokenizer utilities for PhoBERT
Module tiện ích cho PhoBERT tokenizer
"""
from transformers import AutoTokenizer
from typing import List, Dict, Optional
import torch


def load_tokenizer(model_name: str = "vinai/phobert-base", use_fast: bool = True):
    """
    Load PhoBERT tokenizer
    
    Args:
        model_name: Tên model (mặc định: vinai/phobert-base)
        use_fast: Sử dụng fast tokenizer (khuyên dùng)
    
    Returns:
        Tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast
    )
    
    print(f"✓ Loaded tokenizer: {model_name}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max length: {tokenizer.model_max_length}")
    
    return tokenizer


def encode_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    """
    Encode văn bản sử dụng tokenizer
    
    Args:
        text: Văn bản cần encode
        tokenizer: Tokenizer instance
        max_length: Độ dài tối đa
        padding: Padding strategy
        truncation: Có truncate không
        return_tensors: Format return ('pt', 'tf', 'np', None)
    
    Returns:
        Dictionary chứa encoded inputs
    """
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )
    
    return encoding


def encode_pair(
    text_a: str,
    text_b: str,
    tokenizer,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: str = 'only_second',
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    """
    Encode cặp văn bản (cho QA, sentence pair tasks)
    
    Args:
        text_a: Văn bản thứ nhất (question)
        text_b: Văn bản thứ hai (context)
        tokenizer: Tokenizer instance
        max_length: Độ dài tối đa
        padding: Padding strategy
        truncation: Truncation strategy ('only_second' khuyên dùng cho QA)
        return_tensors: Format return
    
    Returns:
        Dictionary chứa encoded inputs
    """
    encoding = tokenizer(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )
    
    return encoding


def batch_encode(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    """
    Encode batch văn bản
    
    Args:
        texts: Danh sách văn bản
        tokenizer: Tokenizer instance
        max_length: Độ dài tối đa
        padding: Padding strategy
        truncation: Có truncate không
        return_tensors: Format return
    
    Returns:
        Dictionary chứa batch encoded inputs
    """
    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )
    
    return encoding


def decode_tokens(
    token_ids: torch.Tensor,
    tokenizer,
    skip_special_tokens: bool = True
) -> str:
    """
    Decode token IDs thành văn bản
    
    Args:
        token_ids: Token IDs (tensor hoặc list)
        tokenizer: Tokenizer instance
        skip_special_tokens: Bỏ qua special tokens ([CLS], [SEP], [PAD])
    
    Returns:
        Decoded text
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().numpy().tolist()
    
    text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return text


def tokenize_for_ner(
    tokens: List[str],
    labels: List[int],
    tokenizer,
    max_length: int = 512
) -> Dict:
    """
    Tokenize cho NER task với label alignment
    
    Args:
        tokens: Danh sách tokens (words)
        labels: Danh sách labels tương ứng
        tokenizer: Tokenizer instance
        max_length: Độ dài tối đa
    
    Returns:
        Dictionary chứa encoded inputs và aligned labels
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Align labels với subword tokens
    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of a word
            aligned_labels.append(labels[word_idx] if word_idx < len(labels) else 0)
        else:
            # Continuation subword
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    
    encoding['labels'] = torch.tensor(aligned_labels, dtype=torch.long)
    
    return encoding


def get_special_tokens(tokenizer) -> Dict[str, str]:
    """
    Lấy các special tokens của tokenizer
    
    Args:
        tokenizer: Tokenizer instance
    
    Returns:
        Dictionary chứa special tokens
    """
    return {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'cls_token': tokenizer.cls_token,
        'sep_token': tokenizer.sep_token,
        'mask_token': tokenizer.mask_token,
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token
    }


def truncate_sequence(
    tokens: List,
    max_length: int,
    strategy: str = 'tail'
) -> List:
    """
    Truncate sequence theo strategy
    
    Args:
        tokens: Danh sách tokens
        max_length: Độ dài tối đa
        strategy: 'tail' (giữ đầu), 'head' (giữ cuối), 'middle' (giữ đầu và cuối)
    
    Returns:
        Truncated tokens
    """
    if len(tokens) <= max_length:
        return tokens
    
    if strategy == 'tail':
        return tokens[:max_length]
    elif strategy == 'head':
        return tokens[-max_length:]
    elif strategy == 'middle':
        half = max_length // 2
        return tokens[:half] + tokens[-(max_length - half):]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def calculate_token_length(text: str, tokenizer) -> int:
    """
    Tính số lượng tokens của văn bản
    
    Args:
        text: Văn bản
        tokenizer: Tokenizer instance
    
    Returns:
        Số lượng tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


def split_long_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Chia văn bản dài thành các chunks với overlap
    Hữu ích cho văn bản hợp đồng dài
    
    Args:
        text: Văn bản dài
        tokenizer: Tokenizer instance
        max_length: Độ dài tối đa mỗi chunk
        overlap: Số tokens overlap giữa các chunks
    
    Returns:
        Danh sách text chunks
    """
    # Tokenize toàn bộ text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    stride = max_length - overlap - 2  # -2 for [CLS] and [SEP]
    
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_length - 2]
        
        # Decode về text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if i + max_length >= len(tokens):
            break
    
    return chunks
