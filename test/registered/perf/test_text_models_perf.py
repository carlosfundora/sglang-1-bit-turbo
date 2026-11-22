import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    parse_models,
)

register_cuda_ci(est_time=3600, suite="nightly-perf-text-2-gpu", nightly=True)

PROFILE_DIR = "performance_profiles_text_models"


class TestNightlyTextModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = []
        models_tp1 = parse_models(
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1
        ) + parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1)
        for model_path in models_tp1:
            cls.models.append(ModelLaunchSettings(model_path, tp_size=1))

        models_tp2 = parse_models(
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2
        ) + parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2)
        for model_path in models_tp2:
            cls.models.append(ModelLaunchSettings(model_path, tp_size=2))

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        all_model_succeed = True

        for model_setup in self.models:
            with self.subTest(model=model_setup.model_path):
                results, success, _ = self.runner.run_benchmark_for_model(
                    model_path=model_setup.model_path,
                    batch_sizes=self.batch_sizes,
                    input_lens=self.input_lens,
                    output_lens=self.output_lens,
                    other_args=model_setup.extra_args,
                )

                if not success:
                    all_model_succeed = False

                self.runner.add_report(results)

        self.runner.write_final_report()

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
