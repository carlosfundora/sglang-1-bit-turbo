from __future__ import annotations

import logging
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
    maybe_bigram_convert,
    page_align_keys,
)
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_NAME,
    ComponentData,
    ComponentName,
    FullComponent,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_last_access_time,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


class UnifiedTreeNode:
    counter = 0

    def __init__(self, component_names: list[str]):
        self.children = defaultdict(partial(UnifiedTreeNode, component_names))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.component_names = list(component_names)
        self.component_data = {
            component_name: ComponentData() for component_name in self.component_names
        }
        self.last_access_time = get_last_access_time()
        self.host_value = None
        self.hit_count = 0
        self.lru_prev: dict[str, UnifiedTreeNode | None] = {
            component_name: None for component_name in self.component_names
        }
        self.lru_next: dict[str, UnifiedTreeNode | None] = {
            component_name: None for component_name in self.component_names
        }
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1

    def component(self, name: str) -> ComponentData:
        return self.component_data[name]

    @property
    def full_value(self) -> Optional[torch.Tensor]:
        return self.component(BASE_COMPONENT_NAME).value

    @full_value.setter
    def full_value(self, value: Optional[torch.Tensor]) -> None:
        self.component(BASE_COMPONENT_NAME).value = value

    def component_value(self, name: str) -> Optional[torch.Tensor]:
        return self.component(name).value

    def set_component_value(self, name: str, value: Optional[torch.Tensor]) -> None:
        self.component(name).value = value

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time


class UnifiedLRUList:
    def __init__(self, component_name: str, component_names: list[str]):
        self.component_name = component_name
        self.head = UnifiedTreeNode(component_names)
        self.tail = UnifiedTreeNode(component_names)
        self.head.lru_next[component_name] = self.tail
        self.tail.lru_prev[component_name] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, old_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        component_name = self.component_name
        new_node.lru_prev[component_name] = old_node
        new_node.lru_next[component_name] = old_node.lru_next[component_name]
        old_node.lru_next[component_name].lru_prev[component_name] = new_node
        old_node.lru_next[component_name] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        component_name = self.component_name
        node.lru_prev[component_name].lru_next[component_name] = node.lru_next[
            component_name
        ]
        node.lru_next[component_name].lru_prev[component_name] = node.lru_prev[
            component_name
        ]

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        x = node.lru_prev[self.component_name]
        while x.component(self.component_name).lock_ref > 0:
            x = x.lru_prev[self.component_name]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        x = node.lru_prev[self.component_name]
        while x.component(self.component_name).lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[self.component_name]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentName, type[TreeComponent]] = {
    ComponentName.FULL: FullComponent,
    ComponentName.MAMBA: MambaComponent,
    ComponentName.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
        component_names: tuple[str, ...],
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        self.is_eagle = params.is_eagle
        self.sliding_window_size = params.sliding_window_size

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        self.component_names = [BASE_COMPONENT_NAME, *component_names]
        self.component_order = list(component_names)
        self.components: dict[str, TreeComponent] = {
            name: COMPONENT_REGISTRY[name](self) for name in self.component_names
        }
        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key
        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.component_names}")

    def reset(self) -> None:
        self.root_node = UnifiedTreeNode(self.component_names)
        self.root_node.key = RadixKey([], None)
        self.root_node.full_value = []
        for component_name in self.component_names:
            self.root_node.component(component_name).lock_ref = 1
        self.component_evictable_size_ = {name: 0 for name in self.component_names}
        self.component_protected_size_ = {name: 0 for name in self.component_names}
        self.lru_lists = {
            component_name: UnifiedLRUList(component_name, self.component_names)
            for component_name in self.component_names
        }

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = maybe_bigram_convert(self.is_eagle, key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)

        key, value = maybe_bigram_convert(self.is_eagle, key, value)
        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {name: 0 for name in self.component_names}

        for component in self.components.values():
            component.drive_eviction(params, tracker)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_NAME],
            swa_num_tokens_evicted=tracker.get(ComponentName.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentName.MAMBA, 0),
        )

    def inc_lock_ref(self, node: UnifiedTreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self.components.values():
            result = component.acquire_component_lock(node, result)
        return result

    def dec_lock_ref(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self.components.values():
            component.release_component_lock(node, params)
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self.components.values():
                comp.cleanup_after_caching_req(req, None, None, True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(prev_prefix_len=req.cache_protected_len)

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self.components.values():
                cl = comp.prepare_for_caching_req(
                    req, insert_params, len(token_ids), True
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            # Key convert + page align
            keys = self.key_convert_fn(token_ids)
            keys = page_align_keys(keys, self.page_size)
            page_aligned_len = len(keys)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
            radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )

        # cleanup
        for comp in self.components.values():
            comp.cleanup_after_caching_req(req, result, insert_params, True)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        token_ids = req.fill_ids

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(prev_prefix_len=req.cache_protected_len)
        effective_cache_len = len(token_ids)
        for comp in self.components.values():
            cl = comp.prepare_for_caching_req(req, insert_params, len(token_ids), False)
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self.components.values():
                comp.cleanup_after_caching_req(req, None, insert_params, False)
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        # Key convert + page align
        keys = self.key_convert_fn(token_ids[:effective_cache_len])
        keys = page_align_keys(keys, self.page_size)
        page_aligned_len = len(keys)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock

        # cleanup
        for comp in self.components.values():
            comp.cleanup_after_caching_req(req, result, insert_params, False)

    ## Internal Helper Functions
    def _for_each_component_lru(self, node: UnifiedTreeNode, lru_op):
        for component_name, component in self.components.items():
            if component.node_has_component_data(node):
                lru_op(self.lru_lists[component_name], node)

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, int]:
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        validators = {
            name: component.create_match_validator()
            for name, component in self.components.items()
        }

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(validators[name](node) for name in self.components):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
                value.append(node.full_value)
                _update_best_if_valid(node)
                break
            value.append(child.full_value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return value, best_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: UnifiedTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        node_update = last_node
        for component_name, component in self.components.items():
            self.lru_lists[component_name].reset_node_and_parents_mru(
                node_update, self.root_node, component.node_has_component_data
            )
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        if best_value_len > 0:
            device_indices = torch.cat(value[:best_value_len])
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=last_node,
            last_host_node=last_node,
        )

        for component in self.components.values():
            result = component.finalize_match_result(
                result, params, value, best_value_len
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.component_names)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.full_value = child.full_value[:split_len].clone()

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        child.full_value = child.full_value[split_len:].clone()

        for component in self.components.values():
            component.redistribute_on_node_split(new_node, child)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._for_each_component_lru(new_node, UnifiedLRUList.insert_mru)
        self._for_each_component_lru(child, UnifiedLRUList.insert_mru)
        child.last_access_time = get_last_access_time()
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self._for_each_component_lru(node, UnifiedLRUList.reset_node_mru)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.component_names)
        new_node.parent = parent
        new_node.key = key
        new_node.full_value = value.clone()
        parent.children[self.get_child_key_fn(key)] = new_node
        self.lru_lists[BASE_COMPONENT_NAME].insert_mru(new_node)
        self.component_evictable_size_[BASE_COMPONENT_NAME] += len(value)
        return new_node

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        self._touch_node(node)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            value_slice = value[:prefix_len]
            consumed_from = prefix_len
            for component in self.components.values():
                comp_consumed_from = component.update_component_on_insert_overlap(
                    node,
                    prefix_len,
                    total_prefix_length,
                    value_slice,
                    params,
                )
                consumed_from = min(consumed_from, comp_consumed_from)

            dup_start = max(0, params.prev_prefix_len - total_prefix_length)
            if dup_start < consumed_from:
                self.token_to_kv_pool_allocator.free(
                    value_slice[dup_start:consumed_from]
                )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        is_new_leaf = False
        if len(key):
            if any(
                comp.should_skip_leaf_creation(total_prefix_length, len(key), params)
                for comp in self.components.values()
            ):
                self.token_to_kv_pool_allocator.free(value)
                return InsertResult(prefix_len=total_prefix_length)
            target_node = self._add_new_node(node, key, value)
            is_new_leaf = True
        else:
            target_node = node

        result = InsertResult(prefix_len=total_prefix_length)
        for component in self.components.values():
            component.commit_insert_component_data(
                target_node, is_new_leaf, params, result
            )
        return result

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        is_leaf: bool,
        tracker: dict[str, int],
    ) -> int:
        freed = comp.evict_component(node, is_leaf=is_leaf)
        tracker[comp.name] += freed
        lru = self.lru_lists[comp.name]
        if lru.in_list(node):
            lru.remove_node(node)
        return freed

    def _cascade_evict(
        self, node: UnifiedTreeNode, trigger: TreeComponent, tracker: dict[str, int]
    ):
        is_leaf = len(node.children) == 0
        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self.components.values():
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node):
                    assert node.component(comp.name).lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, is_leaf=is_leaf, tracker=tracker
                    )

        if is_leaf:
            self._remove_leaf_from_parent(node)
            self._iteratively_delete_tombstone_leaf(node, tracker)

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[str, int]
    ):
        """After a leaf is removed, walk up the parent chain and delete
        any ancestor that is leaf node and has lost any component data (tombstoned)."""
        cur = deleted_node.parent
        while cur != self.root_node and len(cur.children) == 0:
            has_tombstone = any(
                not comp.node_has_component_data(cur)
                for comp in self.components.values()
            )
            if not has_tombstone:
                break

            if any(
                cur.component(comp.name).lock_ref > 0
                for comp in self.components.values()
                if comp.node_has_component_data(cur)
            ):
                break

            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, is_leaf=True, tracker=tracker
                    )
            self._remove_leaf_from_parent(cur)
            cur = cur.parent

    ## Other Apis for Usage Checking
    @property
    def cache_req_mamba_pool(self):
        return self.req_to_token_pool.mamba_pool

    def supports_swa(self) -> bool:
        return ComponentName.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentName.MAMBA in self.components

    def full_evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_NAME, 0)

    def full_protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_NAME, 0)

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentName.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentName.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentName.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentName.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            total_size += len(node.full_value)
            for component_name in self.component_order:
                value = node.component_value(component_name)
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                values.append(child.full_value)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(self, component_name: str) -> torch.Tensor:
        if component_name not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_value(component_name)
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentName.MAMBA)

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentName.SWA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_NAME]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for component_name in self.component_order:
            if component_name.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif component_name.is_mamba:
                available_size = self.cache_req_mamba_pool.available_size()
            else:
                available_size = 0
            lines.append(
                f"Available {component_name}: {available_size + self.component_evictable_size_[component_name]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[component_name]})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        for component_name in self.component_names:
            assert self.component_evictable_size_[component_name] >= 0
            assert self.component_protected_size_[component_name] >= 0

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{component_name}={'yes' if node.component_value(component_name) is not None else 'no'}"
                for component_name in self.component_order
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component(BASE_COMPONENT_NAME).lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))


class UnifiedMambaRadixCache(UnifiedRadixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        assert isinstance(params.req_to_token_pool, HybridReqToTokenPool)
        if not params.enable_mamba_extra_buffer:
            assert params.page_size == 1
        super().__init__(
            params,
            component_names=(ComponentName.MAMBA,),
        )


class UnifiedSWARadixCache(UnifiedRadixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(
            params,
            component_names=(ComponentName.SWA,),
        )


def create_unified_radix_cache(
    params: CacheInitParams,
    component_names: Optional[tuple[ComponentName, ...]] = None,
) -> UnifiedRadixCache:
    if component_names is None:
        component_names = tuple(getattr(params, "unified_tree_components", ()) or ())
    if not component_names:
        if isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            component_names = (ComponentName.SWA,)
        elif isinstance(params.req_to_token_pool, HybridReqToTokenPool):
            component_names = (ComponentName.MAMBA,)
        else:
            raise ValueError("Can not infer unified tree components from params.")
    if component_names == (ComponentName.MAMBA,):
        return UnifiedMambaRadixCache(params)
    if component_names == (ComponentName.SWA,):
        return UnifiedSWARadixCache(params)
    return UnifiedRadixCache(params, component_names=component_names)
