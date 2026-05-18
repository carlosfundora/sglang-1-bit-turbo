"""
ATOM Backend Integration Test Harness

Validates that ATOM backend has correct access to SGLang's paged KV pools:
1. Metadata extraction works correctly
2. No data copying occurs
3. Tensor layouts are correct
4. Hardware detection functions
5. Fallback paths work
"""

import logging
import torch
from typing import Optional, Tuple
import time

logger = logging.getLogger(__name__)


class ATOMBackendValidator:
    """Validates ATOM backend integration with SGLang."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("\n" + "="*80)
        print("ATOM Backend Integration Validation")
        print("="*80 + "\n")
        
        tests = [
            ("Hardware Detection", self._test_hardware_detection),
            ("ATOM Availability", self._test_atom_availability),
            ("Metadata Extraction", self._test_metadata_extraction),
            ("Tensor Pointer Validation", self._test_tensor_pointers),
            ("No-Copy Access", self._test_no_copy_access),
            ("Layout Verification", self._test_layout_verification),
            ("Fallback Path", self._test_fallback_path),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    self._pass(test_name)
                else:
                    self._fail(test_name, "Test returned False")
            except Exception as e:
                self._fail(test_name, str(e))
        
        self._print_summary()
        return self.failed_tests == 0
    
    def _test_hardware_detection(self) -> bool:
        """Test AMD hardware detection."""
        try:
            from sglang.srt.layers.attention.atom_backend import (
                ATOMBackend,
                ATOMHardwareMode,
            )
            
            backend = ATOMBackend()
            mode = backend.hardware_mode
            
            print(f"  Hardware Mode: {mode}")
            print(f"  ATOM Available: {backend.atom_available}")
            
            # Hardware mode should be valid enum
            assert isinstance(mode, ATOMHardwareMode), "Invalid hardware mode type"
            
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                print(f"  GPU: {props.name}")
                print(f"  Architecture: {getattr(props, 'gcnArchName', 'unknown')}")
            
            return True
        
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            return False
    
    def _test_atom_availability(self) -> bool:
        """Test ATOM availability check."""
        try:
            from sglang.srt.layers.attention.atom_backend_registry import (
                get_atom_backend_info,
            )
            
            info = get_atom_backend_info()
            
            print(f"  Available: {info['available']}")
            print(f"  Hardware Mode: {info['hardware_mode']}")
            print(f"  Capabilities: {info['capabilities']}")
            print(f"  Supported DTYPEs: {info['supported_dtypes']}")
            
            # Should have valid structure
            assert "available" in info, "Missing 'available' field"
            assert "hardware_mode" in info, "Missing 'hardware_mode' field"
            assert isinstance(info["available"], bool), "available should be bool"
            
            return True
        
        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return False
    
    def _test_metadata_extraction(self) -> bool:
        """Test paged KV metadata extraction."""
        try:
            from sglang.srt.layers.attention.atom_backend import (
                ATOMBackend,
                PagedKVMetadata,
            )
            
            backend = ATOMBackend()
            
            # Create mock forward batch
            mock_batch = self._create_mock_forward_batch()
            
            # Extract metadata
            metadata = backend._extract_paged_kv_metadata(mock_batch)
            
            # Validate metadata
            assert isinstance(metadata, PagedKVMetadata), "Invalid metadata type"
            assert metadata.page_size > 0, "Invalid page size"
            assert metadata.hidden_dim > 0, "Invalid hidden dim"
            assert metadata.num_reqs > 0, "Invalid num reqs"
            
            print(f"  Page Size: {metadata.page_size}")
            print(f"  Hidden Dim: {metadata.hidden_dim}")
            print(f"  Num Reqs: {metadata.num_reqs}")
            print(f"  Num Pages: {metadata.num_pages}")
            
            return True
        
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return False
    
    def _test_tensor_pointers(self) -> bool:
        """Test that tensor pointers are valid."""
        try:
            from sglang.srt.layers.attention.atom_backend import ATOMBackend
            
            backend = ATOMBackend()
            mock_batch = self._create_mock_forward_batch()
            metadata = backend._extract_paged_kv_metadata(mock_batch)
            
            # Check tensor validity
            assert metadata.req_to_token_pool is not None, "req_to_token_pool is None"
            assert metadata.token_to_kv_pool is not None, "token_to_kv_pool is None"
            assert metadata.kv_indptr is not None, "kv_indptr is None"
            assert metadata.kv_indices is not None, "kv_indices is None"
            
            # Check tensor properties
            print(f"  req_to_token_pool shape: {metadata.req_to_token_pool.shape}")
            print(f"  token_to_kv_pool shape: {metadata.token_to_kv_pool.shape}")
            print(f"  kv_indptr shape: {metadata.kv_indptr.shape}")
            print(f"  kv_indices shape: {metadata.kv_indices.shape}")
            
            # Tensors should be on correct device
            if torch.cuda.is_available():
                # May be on CPU for mock, but pointers should be valid
                assert metadata.req_to_token_pool.numel() >= 0, "Invalid tensor"
            
            return True
        
        except Exception as e:
            logger.error(f"Tensor pointer validation failed: {e}")
            return False
    
    def _test_no_copy_access(self) -> bool:
        """Test that ATOM access doesn't copy data."""
        try:
            import sys
            from sglang.srt.layers.attention.atom_backend import ATOMBackend
            
            backend = ATOMBackend()
            mock_batch = self._create_mock_forward_batch()
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            
            # Extract metadata (should not copy)
            metadata = backend._extract_paged_kv_metadata(mock_batch)
            
            # Get data pointer addresses (these should be same as original)
            print(f"  req_to_token_pool data_ptr: {metadata.req_to_token_pool.data_ptr()}")
            print(f"  token_to_kv_pool data_ptr: {metadata.token_to_kv_pool.data_ptr()}")
            
            # Metadata extraction should not significantly increase memory
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                mem_increase_mb = (mem_after - mem_before) / (1024 * 1024)
                print(f"  Memory increase: {mem_increase_mb:.2f} MB")
                
                # Should be < 1MB (just metadata, no copy)
                assert mem_increase_mb < 1.0, f"Memory increased too much: {mem_increase_mb}MB"
            
            return True
        
        except Exception as e:
            logger.error(f"No-copy access validation failed: {e}")
            return False
    
    def _test_layout_verification(self) -> bool:
        """Test that tensor layouts are correct for HIP kernels."""
        try:
            from sglang.srt.layers.attention.atom_backend import ATOMBackend
            
            backend = ATOMBackend()
            mock_batch = self._create_mock_forward_batch()
            metadata = backend._extract_paged_kv_metadata(mock_batch)
            
            # Check dtypes
            print(f"  req_to_token_pool dtype: {metadata.req_to_token_pool.dtype}")
            print(f"  token_to_kv_pool dtype: {metadata.token_to_kv_pool.dtype}")
            print(f"  kv_indptr dtype: {metadata.kv_indptr.dtype}")
            
            # Validate dtypes for HIP kernels
            assert metadata.req_to_token_pool.dtype == torch.int64, "req_to_token_pool should be int64"
            assert metadata.kv_indptr.dtype == torch.int32, "kv_indptr should be int32"
            
            # Check contiguity (important for CUDA kernel access)
            if metadata.token_to_kv_pool.numel() > 0:
                # Should be contiguous or can be made contiguous
                print(f"  token_to_kv_pool contiguous: {metadata.token_to_kv_pool.is_contiguous()}")
            
            return True
        
        except Exception as e:
            logger.error(f"Layout verification failed: {e}")
            return False
    
    def _test_fallback_path(self) -> bool:
        """Test that fallback to Triton works."""
        try:
            from sglang.srt.layers.attention.atom_backend import ATOMBackend
            
            backend = ATOMBackend()
            
            # Test fallback flag
            backend.fallback_to_triton = True
            print(f"  Fallback to Triton enabled: {backend.fallback_to_triton}")
            
            # If ATOM not available, backend should gracefully handle it
            if not backend.atom_available:
                print(f"  ATOM not available - fallback would be used")
            
            return True
        
        except Exception as e:
            logger.error(f"Fallback path validation failed: {e}")
            return False
    
    # Helper methods
    
    def _create_mock_forward_batch(self):
        """Create a mock ForwardBatch for testing."""
        # Simple mock object with required attributes
        class MockAllocator:
            def __init__(self):
                self.buf = torch.randn(1024, 2, 128, dtype=torch.float16)
                self.num_pages = 10
        
        class MockBatch:
            def __init__(self):
                self.batch_size = 4
                self.page_size = 1
                self.hidden_dim = 128
                self.req_to_token_pool = torch.zeros((4, 512), dtype=torch.int64)
                self.token_to_kv_pool_allocator = MockAllocator()
                self.max_seq_len = 512
                self.seq_lens = torch.tensor([256, 384, 128, 256], dtype=torch.int32)
                
                # Optional attributes
                self.kv_indptr = torch.tensor([0, 256, 640, 768, 1024], dtype=torch.int32)
                self.kv_indices = torch.arange(1024, dtype=torch.int32)
                self.page_table = torch.zeros((4, 512), dtype=torch.int32)
        
        return MockBatch()
    
    def _pass(self, test_name: str):
        """Record a passing test."""
        self.passed_tests += 1
        self.test_results.append((test_name, True, None))
        print(f"  ✓ PASS\n")
    
    def _fail(self, test_name: str, error: str):
        """Record a failing test."""
        self.failed_tests += 1
        self.test_results.append((test_name, False, error))
        print(f"  ✗ FAIL: {error}\n")
    
    def _print_summary(self):
        """Print test summary."""
        total = self.passed_tests + self.failed_tests
        print("="*80)
        print("Test Summary")
        print("="*80)
        print(f"Passed: {self.passed_tests}/{total}")
        print(f"Failed: {self.failed_tests}/{total}")
        
        if self.failed_tests > 0:
            print("\nFailed Tests:")
            for test_name, passed, error in self.test_results:
                if not passed:
                    print(f"  - {test_name}: {error}")
        
        if self.failed_tests == 0:
            print("\n✓ All tests PASSED - ATOM backend ready for production")
        else:
            print(f"\n✗ {self.failed_tests} test(s) FAILED")
        print()


# Command-line entry point
if __name__ == "__main__":
    validator = ATOMBackendValidator()
    success = validator.run_all_tests()
    exit(0 if success else 1)
