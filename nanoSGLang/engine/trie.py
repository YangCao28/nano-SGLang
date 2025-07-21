from typing import Dict, Optional, List

class BlockInfo:
    """存储在 Trie 节点中，指向物理块的元数据"""
    def __init__(self, block_id: int, address: str, full_token_ids: List[int]):
        self.block_id = block_id
        self.status = "IN_GPU"
        self.address = address
        self.full_token_ids = full_token_ids

class TrieNode:
    """Trie 树的节点"""
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        # 只有当一个块在这里结束时，才会有 block_info
        self.block_info: Optional[BlockInfo] = None

class SharedTrie:
    """全局共享的 Trie 树"""
    def __init__(self):
        self.root = TrieNode()

    def get_node(self, token_ids: List[int]) -> Optional[TrieNode]:
        """获取路径末端的节点，如果不存在则返回 None"""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node

    def get_or_create_node(self, token_ids: List[int]) -> TrieNode:
        """获取或创建路径末端的节点"""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        return node
