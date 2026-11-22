import hashlib
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import model_file_verifier as verifier
from sglang.srt.mem_cache.hicache_storage import (
    get_hash_str,
    get_page_hash_values,
    hash_str_to_int64,
)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class TestSglangRustUtils(unittest.TestCase):
    def test_hicache_hash_contract(self):
        expected = hashlib.sha256()
        for token in [1, 2, 3]:
            expected.update(token.to_bytes(4, byteorder="little", signed=False))

        hash_value = get_hash_str([1, 2, 3])
        self.assertEqual(hash_value, expected.hexdigest())
        self.assertEqual(hash_str_to_int64("7fffffffffffffff" + "0" * 48), 2**63 - 1)
        self.assertEqual(hash_str_to_int64("8000000000000000" + "0" * 48), -(2**63))

    def test_hicache_page_hash_contract_with_bigram_tokens(self):
        first_page = hashlib.sha256()
        for token in [11, 12]:
            first_page.update(token.to_bytes(4, byteorder="little", signed=False))
        first_hash = first_page.hexdigest()

        second_page = hashlib.sha256()
        second_page.update(bytes.fromhex(first_hash))
        for token in [13, 14]:
            second_page.update(token.to_bytes(4, byteorder="little", signed=False))

        self.assertEqual(
            get_page_hash_values([(11, 12), (13, 14)], page_size=1),
            [first_hash, second_page.hexdigest()],
        )

    def test_discover_files_preserves_model_verifier_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write(model_dir / "model.safetensors", b"weights")
            _write(model_dir / "config.json", b"{}")
            _write(model_dir / "README.md", b"ignored")
            _write(model_dir / ".hidden", b"ignored")
            _write(model_dir / "nested" / "shard.bin", b"not top level")

            self.assertEqual(
                verifier._discover_files(model_dir),
                ["config.json", "model.safetensors"],
            )

    def test_compute_manifest_from_folder_skips_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write(model_dir / "a.bin", b"abc")
            _write(model_dir / "b.safetensors", b"def")

            manifest = verifier._compute_manifest_from_folder(
                model_path=model_dir,
                filenames=["a.bin", "missing.bin", "b.safetensors"],
                max_workers=2,
            )

            self.assertEqual(set(manifest.files), {"a.bin", "b.safetensors"})
            self.assertEqual(
                manifest.files["a.bin"].sha256,
                hashlib.sha256(b"abc").hexdigest(),
            )
            self.assertEqual(manifest.files["a.bin"].size, 3)

    def test_rust_extension_parity_if_available(self):
        try:
            from sglang.sglang_rust_utils import (
                find_files,
                hicache_hash,
                hicache_hash_to_int64,
                hicache_page_hashes,
                pack_sampling_params,
                saguaro_prefix_hash,
                sha256_manifest,
            )
        except Exception:
            try:
                from sglang_rust_utils import (
                    find_files,
                    hicache_hash,
                    hicache_hash_to_int64,
                    hicache_page_hashes,
                    pack_sampling_params,
                    saguaro_prefix_hash,
                    sha256_manifest,
                )
            except Exception:
                self.skipTest("sglang_rust_utils extension is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write(model_dir / "a.bin", b"abc")
            _write(model_dir / "nested" / "b.bin", b"def")

            files = find_files(str(model_dir))
            self.assertTrue(any(path.endswith("a.bin") for path in files))
            self.assertTrue(any(path.endswith("nested/b.bin") for path in files))

            manifest = sha256_manifest(str(model_dir), ["a.bin", "missing.bin"], 2)
            self.assertEqual(set(manifest), {"a.bin"})
            self.assertEqual(manifest["a.bin"][0], hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(manifest["a.bin"][1], 3)

            self.assertEqual(hicache_hash([1, (2, 3)]), get_hash_str([1, (2, 3)]))
            self.assertEqual(
                hicache_page_hashes([1, 2, 3], 2),
                get_page_hash_values([1, 2, 3], page_size=2),
            )
            self.assertEqual(hicache_hash_to_int64("f" * 64), -1)
            self.assertEqual(
                saguaro_prefix_hash([1, 2, 3], 2),
                hashlib.sha256(b"2,3").hexdigest(),
            )

            class Params:
                def __init__(self, temperature, top_p, top_k, min_p, sampling_seed):
                    self.temperature = temperature
                    self.top_p = top_p
                    self.top_k = top_k
                    self.min_p = min_p
                    self.sampling_seed = sampling_seed

            class Req:
                def __init__(self, params, custom_logit_processor=None):
                    self.sampling_params = params
                    self.custom_logit_processor = custom_logit_processor

            packed = pack_sampling_params(
                [
                    Req(Params(0.0, 1.0, 1, 0.0, None)),
                    Req(Params(0.7, 0.9, 8, 0.1, 123), "processor"),
                ],
                1 << 30,
                True,
                True,
            )
            self.assertAlmostEqual(packed[0][0], 0.0)
            self.assertAlmostEqual(packed[0][1], 0.7)
            self.assertAlmostEqual(packed[1][0], 1.0)
            self.assertAlmostEqual(packed[1][1], 0.9)
            self.assertEqual(packed[2], [1, 8])
            self.assertAlmostEqual(packed[3][0], 0.0)
            self.assertAlmostEqual(packed[3][1], 0.1)
            self.assertEqual(packed[4], [42, 123])
            self.assertEqual(packed[5:], (False, True, True, True, True))


if __name__ == "__main__":
    unittest.main()
