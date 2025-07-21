import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_HEAD': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_HEAD': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_HEAD': 256}, num_warps=8),
    ],
    key=['head_size'],
)
@triton.jit
def copy_kv_prefix_kernel(
    src_ptr, dst_ptr,
    num_tokens_to_copy, num_layers, num_kv_heads, head_size,
    stride_src_l, stride_src_kv, stride_src_h, stride_src_t,
    stride_dst_l, stride_dst_kv, stride_dst_h, stride_dst_t,
    BLOCK_SIZE_HEAD: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_idx = tl.program_id(2)
    for token_idx in range(num_tokens_to_copy):
        for head_offset in range(0, head_size, BLOCK_SIZE_HEAD):
            offsets = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
            mask = offsets < head_size
            src_offset = (layer_idx * stride_src_l + 
                          kv_idx * stride_src_kv + 
                          head_idx * stride_src_h + 
                          token_idx * stride_src_t + 
                          offsets)
            
            dst_offset = (layer_idx * stride_dst_l + 
                          kv_idx * stride_dst_kv + 
                          head_idx * stride_dst_h + 
                          token_idx * stride_dst_t + 
                          offsets)
            data = tl.load(src_ptr + src_offset, mask=mask)
            tl.store(dst_ptr + dst_offset, data, mask=mask)


def copy_kv_prefix_host(
    src_block: torch.Tensor, 
    dst_block: torch.Tensor, 
    num_tokens_to_copy: int
):
    assert src_block.dim() == 5 and dst_block.dim() == 5
    assert src_block.shape == dst_block.shape
    
    num_layers, _, num_heads, block_size, head_size = src_block.shape
    assert num_tokens_to_copy <= block_size
    grid = (num_layers, num_heads, 2)
    copy_kv_prefix_kernel[grid](
        src_block, dst_block,
        num_tokens_to_copy,
        num_layers, num_heads, head_size,
        src_block.stride(0), src_block.stride(1), src_block.stride(2), src_block.stride(3),
        dst_block.stride(0), dst_block.stride(1), dst_block.stride(2), dst_block.stride(3),
    )