# Nano-SGLang

A lightweight SGLang implementation built from scratch.

## 🚀 BlockManager Highlights

- 🌲 **Trie-based Prefix Sharing**  
  Efficiently reuses KV cache blocks for shared token prefixes across sequences.

- ✂️ **Copy-on-Write with Triton Kernel**  
  Cleanly handles divergence by copying only the shared prefix with optimized kernel.

- 🧠 **Two-Level Memory (GPU + Pinned CPU)**  
  Automatically swaps blocks between VRAM and pinned RAM to reduce GPU pressure.

- ♻️ **LRU Eviction Strategy**  
  Frees unused blocks intelligently based on least recently used (LRU) policy.

- 🧱 **Modular Hardware Abstraction Layer**  
  Clean separation of memory management, allocation, and data movement.

- 📦 **Sequence-Friendly Design**  
  Supports streaming with `allocate()`, `can_append()`, `may_append()`, and safe deallocation.

- 🧾 **Debuggable & Transparent**  
  Includes detailed logging and a `print_state()` function for internal visibility.

## Installation

```bash
pip install git+https://github.com/YangCao28/nano-SGLang.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 🔄 BlockManager Allocation Flow

```mermaid
sequenceDiagram
    participant Seq as Sequence
    participant BM as BlockManager
    participant GPU as GPU Memory
    participant CPU as CPU Pinned Memory
    participant Trie as SharedTrie

    Seq->>BM: allocate(seq.tokens)
    BM->>Trie: find longest prefix match
    alt prefix hit & block in GPU
        BM->>GPU: increase ref_count
        BM-->>Seq: share existing block
    else prefix hit & block swapped out (CPU)
        BM->>CPU: swap_in block data
        BM->>GPU: allocate GPU tensor
        BM->>GPU: copy data CPU->GPU
        BM-->>Seq: share block
    else cache miss
        BM->>GPU: allocate new block tensor
        BM->>Trie: create new node with prefix
        BM-->>Seq: new block assigned
    end

    Seq->>BM: deallocate()
    BM->>GPU: decrease ref_count
    alt ref_count == 0
        BM->>GPU: copy block GPU->CPU (swap_out)
        BM->>GPU: free GPU tensor
        BM->>Trie: update block status to SWAPPED_OUT
        BM->>BM: add block_id to free list
    else ref_count > 0
        BM-->>Seq: block still in use
    end

    Note over BM: Periodic LRU eviction triggers
    BM->>BM: select block with ref_count=0 and status IN_GPU
    BM->>GPU: swap_out selected block
    BM->>CPU: copy data GPU->CPU
    BM->>GPU: free GPU tensor
    BM->>BM: mark block as FREE and add to free list
```
```mermaid
graph TD
    BM[BlockManager]

    BM -->|allocate / free block tensors| GPU[GPU Memory Blocks]
    BM -->|allocate / free pinned tensors| CPU[CPU Pinned Memory Blocks]
    BM -->|query / update prefix info| Trie[SharedTrie Prefix Tree]

    GPU -- Copy data --> CPU
    CPU -- Copy data --> GPU

    classDef gpu fill:#a2d2ff,stroke:#000,stroke-width:1px
    classDef cpu fill:#ffafcc,stroke:#000,stroke-width:1px
    classDef trie fill:#cdb4db,stroke:#000,stroke-width:1px
    classDef bm fill:#ffd6a5,stroke:#000,stroke-width:2px,font-weight:bold

    class BM bm
    class GPU gpu
    class CPU cpu
    class Trie trie
```
