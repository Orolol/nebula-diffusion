
import time
import torch
from tqdm import tqdm
from nebula.data import DiffusionTokenizer, create_dataloader

def benchmark_dataloader():
    print("Initializing tokenizer...")
    tokenizer = DiffusionTokenizer()
    
    print("Creating dataloader...")
    # Use same settings as training
    loader = create_dataloader(
        tokenizer=tokenizer,
        batch_size=4,
        max_seq_len=1024,
        num_workers=4,
        prefetch_factor=2
    )
    
    print("Starting benchmark...")
    start_time = time.time()
    total_tokens = 0
    num_batches = 0
    
    # Warmup
    iterator = iter(loader)
    for _ in range(5):
        batch = next(iterator)
    
    benchmark_start = time.time()
    
    try:
        for _ in tqdm(range(50)):
            batch = next(iterator)
            total_tokens += batch["input_ids"].numel()
            num_batches += 1
            
    except StopIteration:
        pass
        
    duration = time.time() - benchmark_start
    tokens_per_sec = total_tokens / duration
    
    print(f"\nResults:")
    print(f"Total tokens: {total_tokens}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_dataloader()
