import torch
from collections import deque, OrderedDict
from typing import List, Dict, Optional
from nanovllm.engine.trie import SharedTrie, TrieNode
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.kernels import copy_kv_prefix_host

class BlockInfo:
    def __init__(self, block_id: int, full_token_ids: List[int]):
        self.block_id, self.status, self.full_token_ids = block_id, "IN_GPU", full_token_ids

class Block:
    def __init__(self, block_id: int, block_shape: tuple, dtype, device):
        self.block_id = block_id
        self.ref_count = 0
        self.status = "FREE"
        self.block_shape = block_shape
        self.dtype = dtype
        self.device = device
        self.gpu_tensor: Optional[torch.Tensor] = None
        self.cpu_tensor: Optional[torch.Tensor] = None
        self.used_tokens = 0

    def has_space(self) -> bool:
        return self.used_tokens < self.block_shape[0]

    def update(self, num_new_tokens: int = 1):
        self.used_tokens += num_new_tokens

    def reset(self):
        self.used_tokens = 0
        self.status = "FREE"
        self.ref_count = 0
        self.gpu_tensor = None
        self.cpu_tensor = None

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, kv_cache_shape_per_token, dtype, device):
        print("\n--- Initializing BlockManager with real PyTorch-based interaction ---")
        self.block_size = block_size
        self.kv_cache_block_shape = (block_size, *kv_cache_shape_per_token)
        self.device = device

        self.trie = SharedTrie()
        self.blocks: List[Block] = [
            Block(i, self.kv_cache_block_shape, dtype, device) for i in range(num_blocks)
        ]
        self.block_to_node_map: Dict[int, TrieNode] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.lru_cache = OrderedDict()

    # --- 硬件交互层 (Hardware Interaction Layer) ---
    def _allocate_gpu_tensor(self, block: Block):
        print(f"    [HW] Allocating GPU tensor for Block {block.block_id} on device {block.device}")
        block.gpu_tensor = torch.empty(block.block_shape, dtype=block.dtype, device=block.device)

    def _free_gpu_tensor(self, block: Block):
        print(f"    [HW] Freeing GPU tensor for Block {block.block_id}")
        block.gpu_tensor = None

    def _allocate_cpu_pinned_tensor(self, block: Block):
        print(f"    [HW] Allocating CPU pinned tensor for Block {block.block_id}")
        block.cpu_tensor = torch.empty(block.block_shape, dtype=block.dtype).pin_memory()

    def _free_cpu_tensor(self, block: Block):
        print(f"    [HW] Freeing CPU tensor for Block {block.block_id}")
        block.cpu_tensor = None

    def _copy_gpu_to_cpu(self, block: Block):
        print(f"    [HW] Copying Block {block.block_id}: GPU VRAM -> CPU RAM")
        if block.gpu_tensor is not None and block.cpu_tensor is not None:
            block.cpu_tensor.copy_(block.gpu_tensor)

    def _copy_cpu_to_gpu(self, block: Block):
        print(f"    [HW] Copying Block {block.block_id}: CPU RAM -> GPU VRAM")
        if block.gpu_tensor is not None and block.cpu_tensor is not None:
            block.gpu_tensor.copy_(block.cpu_tensor)

    def _allocate_new_block(self) -> Block:
        if not self.free_block_ids:
            self._evict_least_used_block()
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.status = "IN_GPU"
        block.ref_count = 1
        self._allocate_gpu_tensor(block)
        self._allocate_cpu_pinned_tensor(block)
        self.lru_cache[block_id] = None
        return block

    def _evict_least_used_block(self):
        print("[BlockManager] LRU eviction triggered. Current LRU order:", list(self.lru_cache.keys()))
        found = False
        for block_id in self.lru_cache:
            block = self.blocks[block_id]
            print(f"  Block {block_id}: ref_count={block.ref_count}, status={block.status}")
            if block.ref_count == 0 and block.status != "SWAPPED_OUT":
                print(f"  -> Evicting Block {block_id}")
                self.swap_out(block_id)
                found = True
                break
        if not found:
            print("[BlockManager] ERROR: No evictable block found! All blocks are either in use or already swapped out.")
            raise RuntimeError("No evictable block found")
    def _can_swap_out_for_allocation(self, num_blocks_needed=1) -> bool:
        evictable_blocks = sum(
            1 for b in self.blocks if b.ref_count == 0 and b.status != "SWAPPED_OUT"
        )
        return (len(self.free_block_ids) + evictable_blocks) >= num_blocks_needed

    def swap_in(self, block_id: int):
        block = self.blocks[block_id]
        self._allocate_gpu_tensor(block)
        self._copy_cpu_to_gpu(block)
        block.status = "IN_GPU"

    def swap_out(self, block_id: int):
        block = self.blocks[block_id]
        print(f"  -> Swapping out Block {block_id}")
        self._copy_gpu_to_cpu(block)
        self._free_gpu_tensor(block)
        block.status = "SWAPPED_OUT"

        node = self.block_to_node_map.get(block_id)
        if node and node.block_info:
            node.block_info.status = "SWAPPED_OUT"

        if block.ref_count == 0:
            print(f"  -> Block {block_id} ready for reuse.")
            self.lru_cache.pop(block_id, None)
            self.free_block_ids.append(block_id)
            block.status = "FREE"


    def touch_block(self, block_id: int):
        if block_id in self.lru_cache:
            self.lru_cache.move_to_end(block_id)

    def allocate(self, seq: Sequence):
        """
        为序列分配或复用Block。这是核心的对外接口，封装了所有复杂性。
        """
        print(f"\nAllocating for Sequence '{seq.seq_id}' with tokens {seq.token_ids}")
        processed_tokens_len = 0
        
        while processed_tokens_len < len(seq.token_ids):
            current_prefix = seq.token_ids[:processed_tokens_len]
            longest_match_node = None
            
            temp_node = self.trie.get_node(current_prefix) or self.trie.root
            for i in range(processed_tokens_len, len(seq.token_ids)):
                token = seq.token_ids[i]
                if token in temp_node.children:
                    temp_node = temp_node.children[token]
                    if temp_node.block_info:
                        longest_match_node = temp_node
                else:
                    break
            if longest_match_node:
                cached_block_info = longest_match_node.block_info
                if cached_block_info.status == "SWAPPED_OUT":
                    self.swap_in(cached_block_info.block_id)
                cached_len = len(cached_block_info.full_token_ids)
                is_pure_prefix = (len(seq.token_ids) >= cached_len and 
                                  seq.token_ids[:cached_len] == cached_block_info.full_token_ids)

                if is_pure_prefix:
                    cached_block = self.blocks[cached_block_info.block_id]
                    print(f"  -> Clean cache hit! Sharing Block {cached_block.block_id} for prefix {cached_block_info.full_token_ids}.")
                    cached_block.ref_count += 1
                    self.touch_block(cached_block.block_id)
                    
                    seq.block_table.append(cached_block.block_id)
                    processed_tokens_len = len(cached_block_info.full_token_ids)
                else:
                    # ========================= TRITON KERNEL INTEGRATION =========================
                    print("  -> Block is contaminated. Triggering CoW with Triton Kernel.")
                    
                    src_block = self.blocks[cached_block_info.block_id]
                    dst_block = self._allocate_new_block()
                    fork_len = 0
                    min_len = min(len(seq.token_ids), cached_len)
                    for k in range(processed_tokens_len, min_len):
                        if seq.token_ids[k] == cached_block_info.full_token_ids[k]:
                            fork_len += 1
                        else:
                            break
                    
                    num_tokens_to_copy = fork_len

                    print(f"  -> Calling Triton Kernel to copy {num_tokens_to_copy} tokens from Block {src_block.block_id} to {dst_block.block_id}.")
                    copy_kv_prefix_host(
                        src_block.gpu_tensor,
                        dst_block.gpu_tensor,
                        num_tokens_to_copy
                    )
                    
                    new_block_info = BlockInfo(dst_block.block_id, seq.token_ids[:processed_tokens_len + 1])
                    anchor_node = self.trie.get_or_create_node(new_block_info.full_token_ids)
                    anchor_node.block_info = new_block_info
                    self.block_to_node_map[dst_block.block_id] = anchor_node
                    seq.block_table.append(dst_block.block_id)
                    processed_tokens_len = len(new_block_info.full_token_ids)
            else:
                print(f"  -> Cache miss. Allocating new block.")
                new_block = self._allocate_new_block()
                
                start_idx = processed_tokens_len
                end_idx = min(len(seq.token_ids), start_idx + self.block_size)
                tokens_for_this_block = seq.token_ids[start_idx:end_idx]
                
                print(f"  -> Allocated New Block {new_block.block_id} for tokens.")
                
                full_prefix_path = seq.token_ids[:end_idx]
                anchor_node = self.trie.get_or_create_node(full_prefix_path)
                anchor_node.block_info = BlockInfo(new_block.block_id, full_prefix_path)
                self.block_to_node_map[new_block.block_id] = anchor_node
                
                seq.block_table.append(new_block.block_id)
                processed_tokens_len = len(full_prefix_path)

    def deallocate(self, seq: Sequence):
        print(f"\nDeallocating for Sequence '{seq.seq_id}'")
        if not seq.block_table: 
            return

        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            print(f"  -> Decremented ref_count for Block {block_id}. New count: {block.ref_count}.")

            if block.ref_count == 0:
                self.swap_out(block_id)
        seq.block_table.clear()

    def print_state(self):
        print("\n--- Current System State ---")
        active_blocks = sorted([b for b in self.blocks if b.status != "FREE"], key=lambda b: b.block_id)
        if not active_blocks:
            print("All blocks are free.")
        for block in active_blocks:
            node = self.block_to_node_map.get(block.block_id)
            prefix = node.block_info.full_token_ids if node and node.block_info else "N/A"
            print(f"Block {block.block_id}: Status={block.status}, RefCount={block.ref_count}, Prefix={prefix}")
        
        print(f"LRU Order (Oldest -> Newest): {list(self.lru_cache.keys())}")
        print("--------------------------")

    def can_append(self, seq: Sequence) -> bool:
        if seq.status != SequenceStatus.RUNNING:
            return False
        last_block_id = seq.block_table[-1] if seq.block_table else None
        if last_block_id is not None:
            block = self.blocks[last_block_id]
            if block.has_space():
                return True
        return bool(self.free_block_ids) or self._can_swap_out_for_allocation()
    
    def can_allocate(self, seq: Sequence) -> bool:
        num_blocks_needed = (len(seq.token_ids) + self.block_size - 1) // self.block_size
        num_free = len(self.free_block_ids)
        num_evictable = sum(
            1 for b in self.blocks if b.ref_count == 0 and b.status == "Free"
        )
        return (num_free + num_evictable) >= num_blocks_needed
    def may_append(self, seq: Sequence) -> None:
        last_block_id = seq.block_table[-1] if seq.block_table else None
        if last_block_id is not None:
            block = self.blocks[last_block_id]
            if block.has_space():
                block.ref_count += 1
                return
        new_block = self._allocate_new_block()
        new_block.ref_count = 1
        new_block.status = "IN_GPU"
        seq.block_table.append(new_block.block_id)

