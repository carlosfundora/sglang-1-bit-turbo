from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentName,
    TreeComponent,
    get_last_access_time,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache_components.unified_radix_cache import (
        UnifiedTreeNode,
    )


class MambaComponent(TreeComponent):
    @property
    def name(self) -> ComponentName:
        return ComponentName.MAMBA

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        name = self.name
        return lambda node: node.component_value(name) is not None

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req
        last_node = result.last_device_node

        if len(value_chunks) > best_value_len:
            chunk_size = get_global_server_args().mamba_cache_chunk_size
            aligned_seqlen = (
                sum(len(v) for v in value_chunks) // chunk_size
            ) * chunk_size
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None
        else:
            branching_seqlen = None

        mamba_value = last_node.component_value(self.name)
        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                    self.cache.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            return
        if node.component_value(self.name) is None:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            node.last_access_time = get_last_access_time()
            return
        self.cache.lru_lists[self.name].reset_node_mru(node)
        node.last_access_time = get_last_access_time()
        result.mamba_exist = True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.set_component_value(self.name, None)
        new_parent.component(self.name).lock_ref = 0

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        freed = len(value)
        self.cache.component_evictable_size_[self.name] -= freed
        if not is_leaf:
            node.set_component_value(self.name, None)
        return freed

    def drive_eviction(self, params: EvictParams, tracker: dict[str, int]) -> None:
        request = params.mamba_num
        lru = self.cache.lru_lists[self.name]
        x = lru.get_lru_no_lock()
        while tracker[self.name] < request and x is not None and lru.in_list(x):
            assert x.component_value(self.name) is not None
            if len(x.children) > 0:
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=False, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next
            else:
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = lru.get_lru_no_lock()

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        value = node.component_value(self.name)
        if value is not None:
            if node.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(value)
                self.cache.component_protected_size_[self.name] += len(value)
            node.component(self.name).lock_ref += 1
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        value = node.component_value(self.name)
        if value is not None:
            assert node.component(self.name).lock_ref > 0
            if node.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(value)
                self.cache.component_protected_size_[self.name] -= len(value)
            node.component(self.name).lock_ref -= 1

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        cache_len = (
            req.mamba_last_track_seqlen
            if self.cache.enable_mamba_extra_buffer
            else token_ids_len
        )
        if is_finished:
            if cache_len is None:
                cache_len = 0
            if self.cache.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            insert_params.mamba_value = mamba_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            if self.cache.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                mamba_value = self.cache.req_to_token_pool.get_mamba_indices(
                    req.req_pool_idx
                ).unsqueeze(-1)
            mamba_value_forked = self.cache.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            if mamba_value_forked is None:
                self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                mamba_value_forked = self.cache.req_to_token_pool.mamba_pool.fork_from(
                    mamba_value
                )
                assert mamba_value_forked is not None, "Can not alloc mamba cache"
            insert_params.mamba_value = mamba_value_forked
            return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        insert_result: Optional[InsertResult],
        insert_params: Optional[InsertParams],
        is_finished: bool,
    ) -> None:
        if is_finished:
            mamba_exist = (
                insert_result.mamba_exist if insert_result is not None else True
            )
            if self.cache.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            else:
                keep_idx = None
            if mamba_exist:
                keep_idx = None
            free_mamba_cache = (
                True if self.cache.enable_mamba_extra_buffer else mamba_exist
            )
            if free_mamba_cache:
                self.cache.req_to_token_pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx
                )
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self.cache.req_to_token_pool.mamba_pool.free(insert_params.mamba_value)
            req.mamba_last_track_seqlen = None
