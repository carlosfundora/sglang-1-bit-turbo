from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentName,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache_components.unified_radix_cache import UnifiedTreeNode


class FullComponent(TreeComponent):
    @property
    def name(self) -> ComponentName:
        return ComponentName.FULL

    def node_has_component_data(self, node: UnifiedTreeNode) -> bool:
        # Override so _for_each_component_lru includes Full in LRU operations
        return node.full_value is not None

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component(self.name).lock_ref = child.component(self.name).lock_ref

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        self.cache.token_to_kv_pool_allocator.free(node.full_value)
        freed = len(node.full_value)
        self.cache.component_evictable_size_[self.name] -= freed
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(self, params: EvictParams, tracker: dict[str, int]) -> None:
        request = params.num_tokens
        lru = self.cache.lru_lists[self.name]
        while tracker[self.name] < request:
            x = lru.get_leaf_lru_no_lock()
            if x is None:
                break
            self.cache._evict_component_and_detach_lru(
                x, self, is_leaf=True, tracker=tracker
            )
            self.cache._cascade_evict(x, self, tracker)

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        cur = node
        while cur != self.cache.root_node:
            if cur.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(cur.full_value)
                self.cache.component_protected_size_[self.name] += len(cur.full_value)
            cur.component(self.name).lock_ref += 1
            cur = cur.parent
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        cur = node
        while cur != self.cache.root_node:
            assert cur.component(self.name).lock_ref > 0
            if cur.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(cur.full_value)
                self.cache.component_protected_size_[self.name] -= len(cur.full_value)
            cur.component(self.name).lock_ref -= 1
            cur = cur.parent
