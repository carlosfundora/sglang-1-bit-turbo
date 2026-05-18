import hashlib
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import model_file_verifier as verifier


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class TestSglangRustUtils(unittest.TestCase):
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
            from sglang.sglang_rust_utils import find_files, sha256_manifest
        except Exception:
            try:
                from sglang_rust_utils import find_files, sha256_manifest
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


if __name__ == "__main__":
    unittest.main()
