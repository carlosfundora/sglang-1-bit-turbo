from __future__ import annotations

import numpy as np

PRISM_Q1_0 = 42
PRISM_Q1_0_G128 = 43
PRISM_DISK_TYPE_REMAP = {
    40: PRISM_Q1_0,
    41: PRISM_Q1_0_G128,
}
PRISM_TYPE_NAMES = {
    PRISM_Q1_0: "Q1_0",
    PRISM_Q1_0_G128: "Q1_0_G128",
}


def gguf_type_name(qtype: int | object) -> str:
    name = getattr(qtype, "name", None)
    if name is not None:
        return name
    try:
        qtype_int = int(qtype)  # type: ignore[arg-type]
    except Exception:
        return str(qtype)
    return PRISM_TYPE_NAMES.get(qtype_int, str(qtype_int))


def _dequantize_prism_q1_blocks(blocks: np.ndarray, block_size: int) -> np.ndarray:
    scales = blocks[:, :2].reshape(-1, 2).view(np.float16).astype(np.float32)
    signs = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little")[:, :block_size]
    values = np.where(signs == 1, scales, -scales).astype(np.float32)
    return values


def _dequantize_prism_q1(data: np.ndarray, block_size: int, type_size: int) -> np.ndarray:
    rows = np.ascontiguousarray(data.view(np.uint8))
    if rows.shape[-1] % type_size != 0:
        raise ValueError(
            f"Invalid Prism GGUF byte shape {rows.shape} for block size {block_size}"
        )
    block_count = rows.shape[-1] // type_size
    flat_blocks = rows.reshape(-1, type_size)
    dequant = _dequantize_prism_q1_blocks(flat_blocks, block_size)
    return dequant.reshape(*rows.shape[:-1], block_count * block_size)


class _PrismQ10Compat:
    @classmethod
    def dequantize(cls, tensor: np.ndarray) -> np.ndarray:
        return _dequantize_prism_q1(tensor, block_size=32, type_size=6)


class _PrismQ10G128Compat:
    @classmethod
    def dequantize(cls, tensor: np.ndarray) -> np.ndarray:
        return _dequantize_prism_q1(tensor, block_size=128, type_size=18)


def ensure_prism_gguf_compat() -> None:
    try:
        import gguf
        import gguf.gguf_reader as gguf_reader
        import gguf.quants as gguf_quants
    except ImportError:
        return

    if getattr(gguf, "_sglang_prism_gguf_compat", False):
        return

    gguf.GGML_QUANT_SIZES[PRISM_Q1_0] = (32, 6)
    gguf.GGML_QUANT_SIZES[PRISM_Q1_0_G128] = (128, 18)
    gguf_quants.GGML_QUANT_SIZES[PRISM_Q1_0] = (32, 6)
    gguf_quants.GGML_QUANT_SIZES[PRISM_Q1_0_G128] = (128, 18)
    gguf_quants._type_traits[PRISM_Q1_0] = _PrismQ10Compat
    gguf_quants._type_traits[PRISM_Q1_0_G128] = _PrismQ10G128Compat

    original_build_tensors = gguf_reader.GGUFReader._build_tensors

    def _build_tensors_with_prism_remap(self, start_offs, fields) -> None:
        tensors = []
        tensor_names = set()
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding="utf-8")
            if tensor_name in tensor_names:
                raise ValueError(f"Found duplicated tensor with name {tensor_name}")
            tensor_names.add(tensor_name)

            raw_qtype = int(raw_dtype[0])
            ggml_type = PRISM_DISK_TYPE_REMAP.get(raw_qtype, raw_qtype)
            n_elems = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            block_size, type_size = gguf.GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(start_offs + offset_tensor[0])

            if ggml_type == gguf.GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            elif ggml_type == gguf.GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == gguf.GGMLQuantizationType.F64:
                item_count = n_elems
                item_type = np.float64
            elif ggml_type == gguf.GGMLQuantizationType.I8:
                item_count = n_elems
                item_type = np.int8
            elif ggml_type == gguf.GGMLQuantizationType.I16:
                item_count = n_elems
                item_type = np.int16
            elif ggml_type == gguf.GGMLQuantizationType.I32:
                item_count = n_elems
                item_type = np.int32
            elif ggml_type == gguf.GGMLQuantizationType.I64:
                item_count = n_elems
                item_type = np.int64
            else:
                item_count = n_bytes
                item_type = np.uint8
                np_dims = gguf_reader.quant_shape_to_byte_shape(np_dims, ggml_type)

            tensors.append(
                gguf_reader.ReaderTensor(
                    name=tensor_name,
                    tensor_type=ggml_type,
                    shape=dims,
                    n_elements=n_elems,
                    n_bytes=n_bytes,
                    data_offset=data_offs,
                    data=self._get(data_offs, item_type, item_count).reshape(np_dims),
                    field=field,
                )
            )
        self.tensors = tensors

    gguf_reader.GGUFReader._build_tensors = _build_tensors_with_prism_remap
    gguf._sglang_prism_gguf_compat = True
    gguf._sglang_prism_gguf_original_build_tensors = original_build_tensors
