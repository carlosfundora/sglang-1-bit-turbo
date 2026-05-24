# JSONSchema Benchmark Summary

## Command Before Refactor
`python3 bench_jsonschema.py`

## Command After Refactor
`python3 bench_jsonschema_rust.py`

## Timing Before Refactor
32758.32 ms (for 10,000 iterations)

## Timing After Refactor
444.33 ms (for 10,000 iterations)

## Percent Change
~98.6% reduction in time (or a ~73x throughput improvement).

## Notes
The Python `jsonschema` library is extremely slow. Since JSON schema validation happens per-request (or per-tool) in the critical path (such as `OpenAIServingBase._check_request_chat_completion`), migrating to the Rust `jsonschema` crate via PyO3 significantly reduces overhead, especially for requests that feature numerous tools or complex parameter schemas.
