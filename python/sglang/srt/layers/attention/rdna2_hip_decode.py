"""RDNA2 (gfx1030) HIP paged flash-decode attention as a torch op.

Device-resident, paged-KV, fp16 analog of the canonical
`rs_rdna2_kernels::flash_decode`. Operates IN-PLACE on the on-GPU KV cache via
kv_indptr/kv_indices (no host round-trips) so it can replace sglang's Triton decode op.

INTER-BLOCK split-K: the work for one (request, q-head) is split across `parallel_blocks`
blocks — grid = (rows, PB). Each block runs an online softmax over a contiguous chunk of
that request's gathered keys and writes a partial (m, l, acc) to scratch; a second
`combine` kernel merges the PB partials per row. This is the llama.cpp fattn-vec
`parallel_blocks` + `flash_attn_combine_results` design: parallelism scales with seq_k, so
it stays fast at long context (microbench: ~6.6-14.6x over single-block on RX 6700 XT).

Contract (matches sglang triton decode_attention_fwd inputs):
  q          [N, Hq, D]   fp16     k_buffer/v_buffer [T, Hk, D] fp16 (paged)
  o          [N, Hq, D]   fp16 (written)
  kv_indptr  [N+1] int32  kv_indices [total] int32   sm_scale float
GQA: kv_head = q_head / (Hq/Hk). No causal mask (kv_indices is the valid set).
Constraints: D % 32 == 0 and D <= 256.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_DECL = """
void rdna2_hip_paged_decode(
    torch::Tensor& o, torch::Tensor& q, torch::Tensor& k_buffer, torch::Tensor& v_buffer,
    torch::Tensor& kv_indptr, torch::Tensor& kv_indices,
    torch::Tensor& part_m, torch::Tensor& part_l, torch::Tensor& part_acc,
    int64_t parallel_blocks, double sm_scale);
"""

_SRC = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <float.h>

namespace rdna2hd {
constexpr int WARP = 32;
constexpr int MAX_ELEMS = 8;   // D/32 <= 8 -> D <= 256

// Phase 1: grid=(rows, PB). Block (row, pb) handles a contiguous chunk of request
// `row/Hq`'s gathered keys; writes a partial online-softmax (m,l,acc) to scratch.
__global__ __attribute__((amdgpu_flat_work_group_size(32, 32)))
void partial_decode(
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
    const __half* __restrict__ q, const __half* __restrict__ k_buf, const __half* __restrict__ v_buf,
    const int* __restrict__ kv_indptr, const int* __restrict__ kv_indices,
    int Hq, int Hk, int D, float scale, int PB)
{
    const int row = blockIdx.x;
    const int pb  = blockIdx.y;
    const int lane = threadIdx.x;
    const int elems = D / WARP;
    const int req = row / Hq;
    const int hq  = row % Hq;
    const int hkv = hq / (Hq / Hk);
    const long q_base = (long)row * D;

    const int ks = kv_indptr[req];
    const int ke = kv_indptr[req + 1];
    const int n = ke - ks;
    const int chunk = (n + PB - 1) / PB;
    const int cs = ks + pb * chunk;
    int ce = cs + chunk; if (ce > ke) ce = ke;

    float qreg[MAX_ELEMS], acc[MAX_ELEMS];
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS; i++) { qreg[i] = 0.f; acc[i] = 0.f; }
    for (int i = 0; i < elems; i++) qreg[i] = __half2float(q[q_base + lane * elems + i]);
    float m = -FLT_MAX, l = 0.f;

    for (int s = cs; s < ce; s++) {
        const int tok = kv_indices[s];
        const long kv = ((long)tok * Hk + hkv) * D;
        float dot = 0.f;
        for (int i = 0; i < elems; i++) dot += qreg[i] * __half2float(k_buf[kv + lane * elems + i]);
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) dot += __shfl_xor(dot, off);
        float score = dot * scale;
        float mn = fmaxf(m, score), corr = __expf(m - mn), p = __expf(score - mn);
        l = l * corr + p;
        for (int i = 0; i < elems; i++) acc[i] = acc[i] * corr + p * __half2float(v_buf[kv + lane * elems + i]);
        m = mn;
    }
    const long pidx = (long)row * PB + pb;
    if (lane == 0) { pm[pidx] = (ce > cs) ? m : -FLT_MAX; pl[pidx] = l; }
    for (int i = 0; i < elems; i++) pacc[pidx * D + lane * elems + i] = acc[i];
}

// Phase 2: grid=(rows). Merge PB partials -> o[row] (fp16).
__global__ __attribute__((amdgpu_flat_work_group_size(32, 32)))
void combine(__half* __restrict__ o, const float* __restrict__ pm, const float* __restrict__ pl,
             const float* __restrict__ pacc, int D, int PB)
{
    const int row = blockIdx.x, lane = threadIdx.x, elems = D / WARP;
    float mg = -FLT_MAX;
    for (int pb = 0; pb < PB; pb++) mg = fmaxf(mg, pm[(long)row * PB + pb]);
    float lg = 0.f; float acc[MAX_ELEMS];
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS; i++) acc[i] = 0.f;
    for (int pb = 0; pb < PB; pb++) {
        const long pidx = (long)row * PB + pb;
        float f = __expf(pm[pidx] - mg);
        lg += pl[pidx] * f;
        for (int i = 0; i < elems; i++) acc[i] += pacc[pidx * D + lane * elems + i] * f;
    }
    float inv = (lg > 0.f) ? 1.f / lg : 0.f;
    for (int i = 0; i < elems; i++) o[(long)row * D + lane * elems + i] = __float2half(acc[i] * inv);
}
}  // namespace rdna2hd

void rdna2_hip_paged_decode(
    torch::Tensor& o, torch::Tensor& q, torch::Tensor& k_buffer, torch::Tensor& v_buffer,
    torch::Tensor& kv_indptr, torch::Tensor& kv_indices,
    torch::Tensor& part_m, torch::Tensor& part_l, torch::Tensor& part_acc,
    int64_t parallel_blocks, double sm_scale)
{
    const int N = q.size(0), Hq = q.size(1), D = q.size(2), Hk = k_buffer.size(1);
    const int rows = N * Hq, PB = (int)parallel_blocks;
    TORCH_CHECK(D % rdna2hd::WARP == 0 && D <= 32 * rdna2hd::MAX_ELEMS, "D %32 and <=256");
    TORCH_CHECK(Hq % Hk == 0, "Hq multiple of Hk");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "fp16 q/k/v expected");

    auto stream = at::hip::getCurrentHIPStream();
    auto* op = reinterpret_cast<__half*>(o.data_ptr<at::Half>());
    auto* qp = reinterpret_cast<__half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<__half*>(k_buffer.data_ptr<at::Half>());
    auto* vp = reinterpret_cast<__half*>(v_buffer.data_ptr<at::Half>());
    auto* ip = kv_indptr.data_ptr<int>();
    auto* ix = kv_indices.data_ptr<int>();
    auto* pm = part_m.data_ptr<float>();
    auto* pl = part_l.data_ptr<float>();
    auto* pacc = part_acc.data_ptr<float>();
    float scale = (float)sm_scale;

    dim3 g1(rows, PB), b1(rdna2hd::WARP), g2(rows), b2(rdna2hd::WARP);
    rdna2hd::partial_decode<<<g1, b1, 0, stream>>>(pm, pl, pacc, qp, kp, vp, ip, ix, Hq, Hk, D, scale, PB);
    rdna2hd::combine<<<g2, b2, 0, stream>>>(op, pm, pl, pacc, D, PB);
}
"""

_mod = None


def _get_module():
    global _mod
    if _mod is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load_inline

        _mod = load_inline(
            name="rdna2_hip_paged_decode_ib",
            cpp_sources=_DECL,
            cuda_sources=_SRC,
            functions=["rdna2_hip_paged_decode"],
            extra_cuda_cflags=["--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64"],
            verbose=False,
        )
        logger.info("RDNA2 HIP paged flash-decode (inter-block split-K): compiled")
    except Exception as e:  # pragma: no cover
        logger.warning(f"RDNA2 HIP paged flash-decode compile failed: {e}")
        _mod = None
    return _mod


def is_available() -> bool:
    return _get_module() is not None


def _choose_parallel_blocks(rows: int, max_kv_len: int, cu: int = 20) -> int:
    """More blocks per query when rows are few / context is long; cap so each chunk
    keeps >=64 keys and total blocks ~fill the GPU."""
    pb = 1
    target = cu * 24
    while pb < 32 and rows * (pb * 2) <= target and max_kv_len >= 64 * (pb * 2):
        pb *= 2
    return pb


def paged_decode(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, sm_scale, max_kv_len=None):
    """q [N,Hq,D] fp16, paged KV via kv_indptr/kv_indices -> writes o [N,Hq,D] fp16."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("rdna2_hip_paged_decode unavailable")
    N, Hq, D = q.shape
    rows = N * Hq
    if max_kv_len is None:
        # cheap: largest per-request key count from kv_indptr
        ind = kv_indptr.to(torch.int64)
        max_kv_len = int((ind[1:] - ind[:-1]).max().item()) if N > 0 else 0
    pb = _choose_parallel_blocks(rows, max_kv_len)

    part_m = torch.empty((rows * pb,), dtype=torch.float32, device=q.device)
    part_l = torch.empty((rows * pb,), dtype=torch.float32, device=q.device)
    part_acc = torch.empty((rows * pb * D,), dtype=torch.float32, device=q.device)
    mod.rdna2_hip_paged_decode(
        o.contiguous(),
        q.contiguous(),
        k_buffer,
        v_buffer,
        kv_indptr.to(torch.int32),
        kv_indices.to(torch.int32),
        part_m, part_l, part_acc,
        int(pb),
        float(sm_scale),
    )
    return o
