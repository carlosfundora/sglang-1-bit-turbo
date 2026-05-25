import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.launch_server import setup_atom_backend_with_flags


class TestAtomLaunchSetup(unittest.TestCase):
    @patch("sglang.launch_server.register_atom_backend")
    def test_wave32_routes_decode_and_prefill(self, _register_atom_backend):
        args = SimpleNamespace(
            attention_backend="atom",
            decode_attention_backend=None,
            prefill_attention_backend=None,
            atom_wave32=True,
            atom_fallback_to_triton=True,
        )

        updated = setup_atom_backend_with_flags(args)

        self.assertEqual(updated.attention_backend, "atom_hybrid")
        self.assertEqual(updated.decode_attention_backend, "atom")
        self.assertEqual(updated.prefill_attention_backend, "triton")

    @patch("sglang.launch_server.register_atom_backend")
    def test_no_fallback_keeps_atom_prefill(self, _register_atom_backend):
        args = SimpleNamespace(
            attention_backend="atom",
            decode_attention_backend=None,
            prefill_attention_backend=None,
            atom_wave32=True,
            atom_fallback_to_triton=False,
        )

        updated = setup_atom_backend_with_flags(args)

        self.assertEqual(updated.prefill_attention_backend, "atom")

    @patch("sglang.launch_server.register_atom_backend")
    def test_non_atom_backend_is_unchanged(self, register_atom_backend):
        args = SimpleNamespace(
            attention_backend="triton",
            decode_attention_backend=None,
            prefill_attention_backend=None,
            atom_wave32=True,
            atom_fallback_to_triton=True,
        )

        updated = setup_atom_backend_with_flags(args)

        self.assertEqual(updated.attention_backend, "triton")
        self.assertIsNone(updated.decode_attention_backend)
        self.assertIsNone(updated.prefill_attention_backend)
        register_atom_backend.assert_not_called()

    @patch("sglang.launch_server.register_atom_backend")
    def test_transformers_model_impl_skips_atom_auto_routing(self, _register_atom_backend):
        args = SimpleNamespace(
            attention_backend="atom",
            decode_attention_backend=None,
            prefill_attention_backend=None,
            atom_wave32=True,
            atom_fallback_to_triton=True,
            model_impl="transformers",
        )

        updated = setup_atom_backend_with_flags(args)

        self.assertEqual(updated.attention_backend, "atom")
        self.assertIsNone(updated.decode_attention_backend)
        self.assertIsNone(updated.prefill_attention_backend)

    @patch("sglang.launch_server.register_atom_backend")
    def test_audio_models_skip_atom_auto_routing(self, _register_atom_backend):
        args = SimpleNamespace(
            attention_backend="atom",
            decode_attention_backend=None,
            prefill_attention_backend=None,
            atom_wave32=True,
            atom_fallback_to_triton=True,
            model_impl="auto",
            get_model_config=lambda: SimpleNamespace(
                is_audio_understandable_model=True,
                is_audio_model=True,
            ),
        )

        updated = setup_atom_backend_with_flags(args)

        self.assertEqual(updated.attention_backend, "atom")
        self.assertIsNone(updated.decode_attention_backend)
        self.assertIsNone(updated.prefill_attention_backend)
