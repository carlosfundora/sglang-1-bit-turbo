import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    flush_cache_with_retry,
    popen_launch_server,
)

register_cuda_ci(est_time=900, suite="stage-c-test-2gpu-h200")

# Mamba cache parameters (must match server --mamba-track-interval / chunk_size)
MAMBA_CACHE_CHUNK_SIZE = 64
MAMBA_TRACK_INTERVAL = 16


class HybridReplayKLMixinBase:
    model = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    kl_div_thres = None
    max_samples = None
    prefill_max_new_tokens = None
    decode_max_new_tokens = None
    replay_batch_size = 1
    server_port_start = 32000
    other_args = []

    @classmethod
    def setUpClass(cls):
        port = find_available_port(cls.server_port_start)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
        )
        assert flush_cache_with_retry(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
            time.sleep(2.0)

    @classmethod
    def _acc_thresholds(cls):
        return {cls.model: {"kl_div": cls.kl_div_thres}}

    @classmethod
    def _expected_prefill_cached_tokens(cls, prefix_len: int) -> int:
        return prefix_len

    @classmethod
    def _expected_decode_cached_tokens(cls, history_len: int, output_len: int) -> int:
        return history_len + output_len

    def flush_cache(self):
        assert flush_cache_with_retry(self.base_url)

    def _assert_cache_hit(self, result: dict, min_cached_tokens: int, label: str):
        cached_tokens = result["meta_info"]["cached_tokens"]
        self.assertGreaterEqual(
            cached_tokens,
            min_cached_tokens,
            f"{label}: expected real prefix-cache hit, got cached_tokens={cached_tokens}",
        )

    def _assert_exact_cached_tokens(
        self, result: dict, expected_cached_tokens: int, label: str
    ):
        cached_tokens = result["meta_info"]["cached_tokens"]
        self.assertEqual(
            cached_tokens,
            expected_cached_tokens,
            f"{label}: expected cached_tokens={expected_cached_tokens}, got {cached_tokens}",
        )

    def _assert_prefill_cached_tokens_near_expected(
        self, result: dict, expected_upper_bound: int, label: str
    ):
        cached_tokens = result["meta_info"]["cached_tokens"]
        expected_lower_bound = max(0, expected_upper_bound - MAMBA_CACHE_CHUNK_SIZE)
        self.assertGreaterEqual(
            cached_tokens,
            expected_lower_bound,
            f"{label}: expected cached_tokens >= {expected_lower_bound}, got {cached_tokens}",
        )
        self.assertLessEqual(
            cached_tokens,
            expected_upper_bound,
            f"{label}: expected cached_tokens <= {expected_upper_bound}, got {cached_tokens}",
        )

    def _get_input_logprobs_batched(
        self, new_input_ids: list[list[int]], output_logprobs: list[list[float]]
    ) -> list[list[float]]:
        input_logprobs = []
        for start in range(0, len(new_input_ids), self.replay_batch_size):
            end = start + self.replay_batch_size
            input_logprobs.extend(
                _get_input_logprobs(
                    self.base_url,
                    new_input_ids[start:end],
                    output_logprobs[start:end],
                )
            )
        return input_logprobs

    def _compare_replay_kl(
        self,
        *,
        label: str,
        replay_input_ids: list[list[int]],
        output_logprobs: list[list[float]],
    ) -> None:
        input_logprobs = self._get_input_logprobs_batched(
            replay_input_ids, output_logprobs
        )
        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            self._acc_thresholds(),
            self.model,
            label,
        )

    @staticmethod
    def _build_replay_item(
        prompt_input_ids: list[int], result: dict
    ) -> tuple[list[int], list[float]]:
        return prompt_input_ids + result["output_ids"], _extract_output_logprobs(result)

    def _assert_prefill_result(
        self,
        *,
        result: dict,
        prefix_input_ids: list[int],
        full_input_ids: list[int],
        label: str,
    ) -> None:
        expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(prefix_input_ids)
        )
        self._assert_exact_cached_tokens(result, expected_cached_tokens, label)
        self._assert_cache_hit(result, expected_cached_tokens, label)

    def _assert_decode_result(
        self,
        *,
        result: dict,
        history_input_ids: list[int],
        previous_output_ids: list[int],
        label: str,
    ) -> None:
        expected_cached_tokens = self._expected_decode_cached_tokens(
            len(history_input_ids), len(previous_output_ids)
        )
        self._assert_exact_cached_tokens(result, expected_cached_tokens, label)
        self._assert_cache_hit(result, expected_cached_tokens, label)

    @staticmethod
    def _build_prefill_boundary_inputs(
        input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        trim_lens = [16, 32, 48, 64]
        full_base, full_step, min_prefix = 1700, 24, 256
        prefix_input_ids = []
        full_input_ids = []
        for i, ids in enumerate(input_ids):
            tail_len = trim_lens[i % len(trim_lens)]
            max_full_len = min(len(ids), full_base + i * full_step)
            prefix_len = (
                max(min_prefix, max_full_len - tail_len) // MAMBA_CACHE_CHUNK_SIZE
            ) * MAMBA_CACHE_CHUNK_SIZE
            full_len = prefix_len + tail_len
            prefix_input_ids.append(ids[:prefix_len])
            full_input_ids.append(ids[:full_len])
        return prefix_input_ids, full_input_ids

    @staticmethod
    def _build_decode_branch_inputs(
        input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        base, step, min_suffix = 960, 24, 8
        suffix_lens = [24, 32, 40, 48]
        first_turn_input_ids = []
        second_turn_suffixes = []
        for i, ids in enumerate(input_ids):
            first_turn_len = min(len(ids), base + i * step)
            branch_suffix_len = suffix_lens[i % len(suffix_lens)]
            branch_suffix = ids[first_turn_len : first_turn_len + branch_suffix_len]
            if len(branch_suffix) < min_suffix:
                branch_suffix = ids[:min_suffix]
            first_turn_input_ids.append(ids[:first_turn_len])
            second_turn_suffixes.append(branch_suffix)
        return first_turn_input_ids, second_turn_suffixes

    @staticmethod
    def _build_multiturn_branch_inputs(
        raw_input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        prefix_base, prefix_step = 640, 32
        turn1_lens = [160, 192, 224]
        turn2_lens = [48, 64, 80]
        min_t1, min_t2 = 48, 16
        shared_prefixes = []
        turn1_suffixes = []
        turn2_suffixes = []
        for group_idx in range(6):
            base_ids = raw_input_ids[group_idx * 3]
            prefix_len = min(len(base_ids), prefix_base + group_idx * prefix_step)
            shared_prefix = base_ids[:prefix_len]

            for branch_idx in range(3):
                source_ids = raw_input_ids[group_idx * 3 + branch_idx]
                turn1_start = min(prefix_len, max(128, len(source_ids) // 4))
                turn1_len = turn1_lens[branch_idx] + group_idx * 8
                turn1_suffix = source_ids[turn1_start : turn1_start + turn1_len]
                if len(turn1_suffix) < min_t1:
                    turn1_suffix = source_ids[:min_t1]

                turn2_start = min(
                    turn1_start + turn1_len,
                    max(turn1_start + 16, len(source_ids) // 2),
                )
                turn2_len = turn2_lens[branch_idx]
                turn2_suffix = source_ids[turn2_start : turn2_start + turn2_len]
                if len(turn2_suffix) < min_t2:
                    turn2_suffix = source_ids[-min_t2:]

                shared_prefixes.append(shared_prefix)
                turn1_suffixes.append(turn1_suffix)
                turn2_suffixes.append(turn2_suffix)
        return shared_prefixes, turn1_suffixes, turn2_suffixes

    def _run_prefill_replay_case(
        self,
        *,
        label: str,
        prefix_input_ids: list[list[int]],
        full_input_ids: list[list[int]],
        max_new_tokens: int,
    ):
        self.flush_cache()
        _generate(self.base_url, prefix_input_ids, max_new_tokens=0)

        results = _generate(
            self.base_url,
            full_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(results), len(full_input_ids))

        replay_input_ids = []
        output_logprobs = []
        for i, result in enumerate(results):
            self._assert_prefill_result(
                result=result,
                prefix_input_ids=prefix_input_ids[i],
                full_input_ids=full_input_ids[i],
                label=f"{label}[{i}]",
            )
            replay_item, output_logprobs_item = self._build_replay_item(
                full_input_ids[i], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )

    def _run_decode_replay_case(
        self,
        *,
        label: str,
        first_turn_input_ids: list[list[int]],
        second_turn_suffixes: list[list[int]],
        max_new_tokens: int,
    ):
        self.flush_cache()
        first_turn_results = _generate(
            self.base_url,
            first_turn_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(first_turn_results), len(first_turn_input_ids))

        second_turn_input_ids = [
            first_turn_input_ids[i]
            + first_turn_results[i]["output_ids"]
            + second_turn_suffixes[i]
            for i in range(len(first_turn_input_ids))
        ]

        second_turn_results = _generate(
            self.base_url,
            second_turn_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(second_turn_results), len(second_turn_input_ids))

        replay_input_ids = []
        output_logprobs = []
        for i, result in enumerate(second_turn_results):
            self._assert_decode_result(
                result=result,
                history_input_ids=first_turn_input_ids[i],
                previous_output_ids=first_turn_results[i]["output_ids"],
                label=f"{label}[{i}]",
            )
            replay_item, output_logprobs_item = self._build_replay_item(
                second_turn_input_ids[i], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )

    def _run_multiturn_branching_replay_case(
        self,
        *,
        label: str,
        shared_prefixes: list[list[int]],
        turn1_suffixes: list[list[int]],
        turn2_suffixes: list[list[int]],
        max_new_tokens: int,
    ):
        self.flush_cache()
        _generate(self.base_url, shared_prefixes, max_new_tokens=0)

        turn1_input_ids = [
            shared_prefixes[i] + turn1_suffixes[i] for i in range(len(shared_prefixes))
        ]
        turn1_results = _generate(
            self.base_url,
            turn1_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(turn1_results), len(turn1_input_ids))

        for i, result in enumerate(turn1_results):
            self._assert_prefill_result(
                result=result,
                prefix_input_ids=shared_prefixes[i],
                full_input_ids=turn1_input_ids[i],
                label=f"{label}[turn1][{i}]",
            )

        turn2_input_ids = [
            turn1_input_ids[i] + turn1_results[i]["output_ids"] + turn2_suffixes[i]
            for i in range(len(turn1_input_ids))
        ]

        # Interleave branches from different shared trunks so the tree is exercised
        # under realistic multi-session progression instead of single-session replay.
        interleaved_order = [
            group_idx * 3 + branch_idx
            for branch_idx in range(3)
            for group_idx in range(len(turn2_input_ids) // 3)
        ]

        ordered_turn2_inputs = [turn2_input_ids[i] for i in interleaved_order]
        turn2_results = _generate(
            self.base_url,
            ordered_turn2_inputs,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(turn2_results), len(ordered_turn2_inputs))

        replay_input_ids = []
        output_logprobs = []
        for ordered_idx, result in enumerate(turn2_results):
            original_idx = interleaved_order[ordered_idx]
            self._assert_decode_result(
                result=result,
                history_input_ids=turn1_input_ids[original_idx],
                previous_output_ids=turn1_results[original_idx]["output_ids"],
                label=f"{label}[turn2][{original_idx}]",
            )
            replay_item, output_logprobs_item = self._build_replay_item(
                ordered_turn2_inputs[ordered_idx], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )


class MambaReplayKLMixin(HybridReplayKLMixinBase):
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    kl_div_thres = 0.0025
    max_samples = 50
    prefill_max_new_tokens = 256
    decode_max_new_tokens = 256
    replay_batch_size = 2
    server_port_start = 32000
    other_args = [
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        MAMBA_TRACK_INTERVAL,
        "--enable-unified-radix-tree",
    ]

    @classmethod
    def _expected_prefill_cached_tokens(cls, prefix_len: int) -> int:
        return (prefix_len // MAMBA_CACHE_CHUNK_SIZE) * MAMBA_CACHE_CHUNK_SIZE

    @classmethod
    def _expected_decode_cached_tokens(cls, history_len: int, output_len: int) -> int:
        if output_len <= 0:
            return history_len
        seq_len = history_len + output_len - 1
        return (seq_len // MAMBA_TRACK_INTERVAL) * MAMBA_TRACK_INTERVAL

    def _assert_prefill_result(
        self,
        *,
        result: dict,
        prefix_input_ids: list[int],
        full_input_ids: list[int],
        label: str,
    ) -> None:
        expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(prefix_input_ids)
        )
        self._assert_prefill_cached_tokens_near_expected(
            result,
            expected_cached_tokens,
            label,
        )


class TestHybridMambaReplayKL(MambaReplayKLMixin, unittest.TestCase):
    def test_prefill_replay_kl_across_track_boundaries(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2200,
            num_samples=self.max_samples,
        )[: self.max_samples]
        prefix_input_ids, full_input_ids = self._build_prefill_boundary_inputs(
            input_ids
        )

        self._run_prefill_replay_case(
            label="test_prefill_replay_kl_across_track_boundaries",
            prefix_input_ids=prefix_input_ids,
            full_input_ids=full_input_ids,
            max_new_tokens=self.prefill_max_new_tokens,
        )

    def test_decode_replay_kl_with_branching_suffixes(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=1400,
            num_samples=self.max_samples,
        )[: self.max_samples]
        first_turn_input_ids, second_turn_suffixes = self._build_decode_branch_inputs(
            input_ids
        )

        self._run_decode_replay_case(
            label="test_decode_replay_kl_with_branching_suffixes",
            first_turn_input_ids=first_turn_input_ids,
            second_turn_suffixes=second_turn_suffixes,
            max_new_tokens=self.decode_max_new_tokens,
        )

    def test_multiturn_replay_kl_with_interleaved_abc_branches(self):
        raw_input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2400,
            num_samples=self.max_samples,
        )[: self.max_samples]
        shared_prefixes, turn1_suffixes, turn2_suffixes = (
            self._build_multiturn_branch_inputs(raw_input_ids)
        )

        self._run_multiturn_branching_replay_case(
            label="test_multiturn_replay_kl_with_interleaved_abc_branches",
            shared_prefixes=shared_prefixes,
            turn1_suffixes=turn1_suffixes,
            turn2_suffixes=turn2_suffixes,
            max_new_tokens=self.decode_max_new_tokens,
        )

    def test_prefill_replay_kl_after_competing_branch_pressure(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2600,
            num_samples=self.max_samples + 4,
        )[: self.max_samples + 4]

        common_prefix = input_ids[0][:960]
        target_suffix = input_ids[0][960:1680]
        if len(target_suffix) < 256:
            target_suffix = input_ids[1][:256]
        target_full = common_prefix + target_suffix

        pressure_prompts = []
        for i, ids in enumerate(input_ids[1:3]):
            suffix = ids[960 : 960 + 448 + i * 32]
            if len(suffix) < 256:
                suffix = ids[:256]
            pressure_prompts.append(common_prefix + suffix)

        self.flush_cache()
        _generate(self.base_url, [common_prefix], max_new_tokens=0)

        hot_result = _generate(
            self.base_url,
            [target_full],
            max_new_tokens=self.prefill_max_new_tokens,
            return_logprob=True,
        )[0]
        hot_expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(common_prefix)
        )
        self._assert_exact_cached_tokens(
            hot_result,
            hot_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[hot]",
        )
        self._assert_cache_hit(
            hot_result,
            hot_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[hot]",
        )

        for pressure_prompt in pressure_prompts:
            _generate(
                self.base_url,
                [pressure_prompt],
                max_new_tokens=24,
                return_logprob=False,
            )

        revisit_result = _generate(
            self.base_url,
            [target_full],
            max_new_tokens=self.prefill_max_new_tokens,
            return_logprob=True,
        )[0]
        revisit_expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(target_full)
        )
        self._assert_exact_cached_tokens(
            revisit_result,
            revisit_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[revisit]",
        )
        self._assert_cache_hit(
            revisit_result,
            revisit_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[revisit]",
        )

        replay_input_ids, output_logprobs = zip(
            self._build_replay_item(target_full, revisit_result)
        )
        self._compare_replay_kl(
            label="test_prefill_replay_kl_after_competing_branch_pressure",
            replay_input_ids=list(replay_input_ids),
            output_logprobs=list(output_logprobs),
        )


class SWAReplayKLMixin(HybridReplayKLMixinBase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    kl_div_thres = 0.003
    max_samples = 24
    replay_batch_size = 1
    decode_max_new_tokens = 256
    server_port_start = 33000
    other_args = [
        "--tp-size",
        "2",
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
        "--random-seed",
        "20260322",
        "--enable-unified-radix-tree",
    ]

    @staticmethod
    def _select_stable_inputs(
        input_ids: list[list[int]],
        *,
        target_count: int,
        min_len: int,
        max_len: int,
        target_len: int,
    ) -> list[list[int]]:
        filtered = [ids for ids in input_ids if min_len <= len(ids) <= max_len]
        if len(filtered) < target_count:
            filtered = sorted(input_ids, key=lambda ids: abs(len(ids) - target_len))
        else:
            filtered = sorted(filtered, key=lambda ids: abs(len(ids) - target_len))
        return filtered[:target_count]

    @staticmethod
    def _build_branching_decode_inputs(
        input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        base, step, min_suffix = 896, 16, 16
        suffix_lens = [48, 64, 80, 96]
        first_turn_input_ids = []
        second_turn_suffixes = []
        for i, ids in enumerate(input_ids):
            first_turn_len = min(len(ids), base + i * step)
            suffix_len = suffix_lens[i % len(suffix_lens)]
            second_turn_suffix = ids[first_turn_len : first_turn_len + suffix_len]
            if len(second_turn_suffix) < min_suffix:
                second_turn_suffix = ids[:min_suffix]
            first_turn_input_ids.append(ids[:first_turn_len])
            second_turn_suffixes.append(second_turn_suffix)
        return first_turn_input_ids, second_turn_suffixes


class TestHybridSWAReplayKL(SWAReplayKLMixin, unittest.TestCase):
    def test_decode_replay_kl_with_sliding_window_branching(self):
        raw_input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=1600,
            num_samples=self.max_samples,
        )[: self.max_samples]
        input_ids = self._select_stable_inputs(
            raw_input_ids,
            target_count=self.max_samples,
            min_len=1200,
            max_len=2400,
            target_len=1600,
        )
        first_turn_input_ids, second_turn_suffixes = (
            self._build_branching_decode_inputs(input_ids)
        )

        self._run_decode_replay_case(
            label="test_decode_replay_kl_with_sliding_window_branching",
            first_turn_input_ids=first_turn_input_ids,
            second_turn_suffixes=second_turn_suffixes,
            max_new_tokens=self.decode_max_new_tokens,
        )

    def test_multiturn_replay_kl_with_interleaved_abc_branches(self):
        raw_input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2400,
            num_samples=self.max_samples,
        )[: self.max_samples]
        shared_prefixes, turn1_suffixes, turn2_suffixes = (
            self._build_multiturn_branch_inputs(raw_input_ids)
        )

        self._run_multiturn_branching_replay_case(
            label="test_multiturn_replay_kl_with_interleaved_abc_branches",
            shared_prefixes=shared_prefixes,
            turn1_suffixes=turn1_suffixes,
            turn2_suffixes=turn2_suffixes,
            max_new_tokens=self.decode_max_new_tokens,
        )


if __name__ == "__main__":
    unittest.main()
