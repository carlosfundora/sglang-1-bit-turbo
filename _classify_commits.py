import re, json

commits = []
with open('/tmp/upstream-commits.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sha = line.split()[0]
        msg = line[len(sha):].strip()
        commits.append((sha, msg))

def classify(sha, msg):
    ml = msg.lower()
    
    # === SKIP categories ===
    if any(k in ml for k in ['[asr]', 'whisper', 'qwen3-asr', 'voxtral', 'speech-to-text', 'transcription adapter']):
        return 'SKIP', 'asr_whisper'
    if '[diffusion]' in ml or ('diffusion' in ml and 'diffusion' != ml):
        return 'SKIP', 'diffusion'
    if '[npu]' in ml or 'ascend' in ml:
        return 'SKIP', 'npu'
    if ml.startswith('[ci]') or ml.startswith('[ci ') or ('ci:' in ml[:15] and 'fix' not in ml[:30]):
        return 'SKIP', 'ci'
    if 'trivy' in ml:
        return 'SKIP', 'docker'
    if any(k in ml for k in ['sm103', 'sm120', 'sm75', 'blackwell', 'b200', 'b300', 'nvfp4', 'trtllm', 'h200', 'cutlass nvfp4', '5090']):
        return 'SKIP', 'nvidia_specific'
    if 'flashinfer' in ml and 'disable' not in ml and 'lazy' not in ml:
        return 'SKIP', 'flashinfer'
    if 'flashmla' in ml:
        return 'SKIP', 'flashinfer'
    if any(k in ml for k in ['mi300', 'mi30x', 'mi35x', 'mi355']):
        return 'SKIP', 'amd_mi_specific'
    if 'aiter' in ml:
        return 'SKIP', 'amd_mi_specific'
    if any(k in ml for k in ['[pd]', '[disagg]', 'nixl', 'mooncake', 'disaggregation', 'staging buffer', 'kv transfer', 'kvreceiver']):
        return 'SKIP', 'pd_disagg'
    if any(k in ml for k in ['hicache', 'hisparse', 'himamba', 'hiradix', 'hi-mamba']):
        return 'SKIP', 'hicache_hisparse'
    if 'lora' in ml:
        return 'SKIP', 'lora'
    if 'score api' in ml or 'enginescoremixin' in ml:
        return 'SKIP', 'score_api'
    if 'grpc' in ml:
        return 'SKIP', 'grpc'
    if any(k in ml for k in ['apple silicon', 'mlx', '[mps]']):
        return 'SKIP', 'apple_mps'
    if any(k in ml for k in ['[vlm]', 'lfm2-vl', 'mm-attention']):
        return 'SKIP', 'vlm'
    if any(k in ml for k in ['gemma 4', 'gemma4', 'grok-', 'glm-5', 'glm-4', 'glm-v', 'lfm2', 'minimax',
                              'kimi k2', 'kimi-linear']):
        if 'eagle' not in ml and 'spec' not in ml:
            return 'SKIP', 'unrelated_model'
    if '[mamba]' in ml or 'mamba' in ml:
        return 'SKIP', 'mamba'
    if '[workflow]' in ml:
        return 'SKIP', 'workflow'
    if '[musa]' in ml or 'mate (' in ml:
        return 'SKIP', 'musa'
    if 'gateway' in ml:
        return 'SKIP', 'gateway'
    if any(k in ml for k in ['fa3', 'fa4']) and 'eagle' not in ml:
        return 'SKIP', 'nvidia_specific'
    if '[kda]' in ml:
        return 'SKIP', 'specialized'
    if 'nightly' in ml and ('benchmark' in ml or 'perf' in ml):
        return 'SKIP', 'benchmark'
    if ml.startswith('[benchmark]'):
        return 'SKIP', 'benchmark'
    if any(k in ml for k in ['rerun-test', 'ci_auto_bisect', 'runner label', 'suite detection',
                              'suite validation', 'test coverage report', 'test skills and guide',
                              'ci permission', 'cuda coredumps', 'rerun-ut']):
        return 'SKIP', 'ci'
    if 'ring test' in ml:
        return 'SKIP', 'ci'
    if '[rl]' in ml:
        return 'SKIP', 'rl'
    if '[pp]' in ml:
        return 'SKIP', 'pipeline_parallel'
    if 'kernel release' in ml or 'bump sgl-kernel' in ml or 'bump sglang-kernel' in ml:
        return 'SKIP', 'kernel_release'
    if 'deepep' in ml:
        return 'SKIP', 'specialized'
    if 'indexcache' in ml.replace(' ', '') and 'deepseek' in ml:
        return 'SKIP', 'unrelated_model'
    if 'remove obsolete sgl-kernel' in ml:
        return 'SKIP', 'kernel_release'
    if 'mmdataprocessor' in ml or 'mm_inputs' in ml or 'multimodal' in ml:
        return 'SKIP', 'vlm'
    if 'dflash' in ml:
        return 'SKIP', 'specialized'
    if 'transformers modeling backend' in ml:
        return 'SKIP', 'specialized'
    if 'docker' in ml:
        return 'SKIP', 'docker'
    if 'nsa' in ml:
        return 'SKIP', 'nvidia_specific'
    if 'deepseek v3' in ml.replace('.', ' '):
        return 'SKIP', 'unrelated_model'
    if 'dsa' in ml and ('[dsa]' in ml):
        return 'SKIP', 'nvidia_specific'
    if 'cuda graph' in ml.replace('_', ' '):
        return 'SKIP', 'cuda_graph'
    if 'graph_capture' in ml:
        return 'SKIP', 'cuda_graph'
    if 'piecewise cuda' in ml:
        return 'SKIP', 'cuda_graph'
    if 'pp key' in ml:
        return 'SKIP', 'pipeline_parallel'
    if 'model streamer' in ml or 'runai' in ml:
        return 'SKIP', 'specialized'
    if 'hash utils' in ml and 'hicache' in ml:
        return 'SKIP', 'hicache_hisparse'
    if 'automodel mapping' in ml:
        return 'SKIP', 'vlm'
    if 'registration api' in ml and 'linear attention' in ml:
        return 'SKIP', 'specialized'
    if 'remove duplicate words' in ml:
        return 'SKIP', 'trivial'
    if 'community fa3' in ml:
        return 'SKIP', 'nvidia_specific'
    if '[moe]' in ml or 'moe-a2a' in ml:
        return 'SKIP', 'specialized'
    if 'dp-attention' in ml and 'bench_one_batch' in ml:
        return 'SKIP', 'benchmark'
    if 'run_eval' in ml and ('metrics' in ml or 'sampler' in ml or 'dump_metric' in ml):
        return 'SKIP', 'eval'
    
    # === WANT categories ===
    if 'llama eagle3' in ml or 'eagle3' in ml:
        return 'WANT', 'eagle3_fix'
    if 'eagle_infer' in ml:
        return 'WANT', 'eagle3_fix'
    if '[spec][ngram]' in ml or 'ngram' in ml:
        return 'WANT', 'ngram_improvement'
    if 'spec_v2' in ml or 'spec v2' in ml or 'specv2' in ml:
        return 'WANT', 'spec_v2_fix'
    if 'chained spec' in ml:
        return 'WANT', 'spec_fix'
    if 'multi layer eagle' in ml:
        return 'WANT', 'eagle_fix'
    if 'isolate spec v1' in ml:
        return 'WANT', 'spec_fix'
    if 'draft extend' in ml:
        return 'WANT', 'spec_fix'
    if 'scheduler' in ml and ('fix' in ml or 'update' in ml or 'add' in ml):
        return 'WANT', 'scheduler'
    if 'merge batch' in ml:
        return 'WANT', 'scheduler'
    if 'multi-thread weight loading' in ml or 'multi thread weight' in ml:
        return 'WANT', 'weight_loading'
    if 'remote weight' in ml:
        return 'WANT', 'weight_loading'
    if 'tied embeddings weight' in ml:
        return 'WANT', 'weight_loading'
    if 'tokenizer_manager' in ml and ('fix' in ml or 'pre-init' in ml):
        return 'WANT', 'runtime_fix'
    if 'think_end_id' in ml and 'model_config' in ml:
        return 'WANT', 'model_config'
    if 'think_end_id' in ml and ('fix' in ml or 'grammar' in ml):
        return 'WANT', 'grammar_fix'
    if 'xgrammar' in ml:
        return 'WANT', 'grammar_fix'
    if 'pause_generation' in ml or 'pause generation' in ml:
        return 'WANT', 'runtime_fix'
    if 'weight update locking' in ml or 'update_weights' in ml:
        return 'WANT', 'runtime_fix'
    if 'toctou race' in ml and 'weight update' in ml:
        return 'WANT', 'runtime_fix'
    if 'writer lock deadlock' in ml:
        return 'WANT', 'runtime_fix'
    if 'http 400' in ml or 'streaming validation' in ml:
        return 'WANT', 'api_fix'
    if 'multi tool streaming' in ml:
        return 'WANT', 'api_fix'
    if 'logprob' in ml:
        return 'WANT', 'logprobs'
    if 'streaming logprobs' in ml:
        return 'WANT', 'logprobs'
    if 'topk postprocessing' in ml:
        return 'WANT', 'perf'
    if 'fuse temperature' in ml and 'sampling' in ml:
        return 'WANT', 'sampling_kernel'
    if 'clean loggings' in ml:
        return 'WANT', 'log_cleanup'
    if 'subprocess watchdog' in ml:
        return 'WANT', 'log_cleanup'
    if 'setuptools-scm' in ml:
        return 'WANT', 'build_fix'
    if 'f-string prefix' in ml:
        return 'WANT', 'code_fix'
    if 'get_numa_node' in ml or 'numa_node' in ml:
        return 'WANT', 'log_cleanup'
    if 'req_time_stats' in ml:
        return 'WANT', 'perf'
    if 'added tokens config' in ml:
        return 'WANT', 'tokenizer_fix'
    if 'qwen2_5_math' in ml or ('qwen2' in ml and 'fix' in ml):
        return 'WANT', 'qwen_fix'
    if '--stream-response' in ml:
        return 'WANT', 'server_args'
    if 'server_info' in ml or 'get_server_info' in ml:
        return 'MAYBE', 'api_migration'
    if 'rope_theta' in ml or 'rope theta' in ml:
        return 'WANT', 'rope_fix'
    if 'fused_qknorm_rope' in ml or 'qknorm_rope' in ml:
        return 'WANT', 'jit_kernel'
    if 'repetition_penalty' in ml:
        return 'MAYBE', 'feature'
    
    # === MAYBE categories ===
    if 'reasoning tokens' in ml:
        return 'MAYBE', 'feature'
    if 'dp attention' in ml and ('port' in ml or 'ipv6' in ml):
        return 'MAYBE', 'dp_attention'
    if 'pcg' in ml:
        return 'MAYBE', 'pcg'
    if 'jit' in ml and ('kernel' in ml or 'rmsnorm' in ml):
        return 'MAYBE', 'jit_kernel'
    if '__getitem__' in ml:
        return 'MAYBE', 'runtime'
    if 'parallel state refactor' in ml:
        return 'MAYBE', 'amd_general'
    if 'rocm' in ml:
        return 'MAYBE', 'amd_general'
    if '[amd]' in ml:
        return 'MAYBE', 'amd_general'
    if 'http2' in ml:
        return 'MAYBE', 'feature'
    if 'maxitems' in ml.replace(' ', '').replace('-', ''):
        return 'MAYBE', 'api_fix'
    if 'mistral embedding' in ml:
        return 'MAYBE', 'model_fix'
    
    # Generic catch
    if any(k in ml for k in ['[ci]', '[test]', 'ci ', 'test ']):
        if 'fix' not in ml:
            return 'SKIP', 'ci'
    if '[doc]' in ml:
        return 'SKIP', 'doc'
    if '[misc]' in ml:
        return 'SKIP', 'misc'
    if ml.startswith('revert'):
        return 'SKIP', 'revert'
    
    return 'MAYBE', 'uncategorized'

results = {'WANT': [], 'MAYBE': [], 'SKIP': []}
skip_cats = {}

for sha, msg in commits:
    cat, subcat = classify(sha, msg)
    results[cat].append((sha, msg, subcat))
    if cat == 'SKIP':
        skip_cats[subcat] = skip_cats.get(subcat, 0) + 1

print(f"Total: {len(commits)}")
print(f"WANT: {len(results['WANT'])}")
print(f"MAYBE: {len(results['MAYBE'])}")
print(f"SKIP: {len(results['SKIP'])}")
print()
print("=== WANT ===")
for sha, msg, subcat in results['WANT']:
    print(f"  {sha} [{subcat}] {msg}")
print()
print("=== MAYBE ===")
for sha, msg, subcat in results['MAYBE']:
    print(f"  {sha} [{subcat}] {msg}")
print()
print("=== SKIP categories ===")
for k, v in sorted(skip_cats.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

with open('/mnt/ai/forks/sglang-1-bit-turbo/_classification.json', 'w') as f:
    json.dump(results, f, indent=2)
