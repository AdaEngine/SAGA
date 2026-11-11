# SAGA vs wgpu benchmarks

Both programs run the same WGSL files from `shaders/`, the same problem sizes, and the same
verification, then print one table in the same format. Both force the Vulkan backend by default
so the comparison is runtime-to-runtime on the same driver.

| Case | What it measures |
| --- | --- |
| `compile` | WGSL → pipeline creation, 20 shader variants (cold, cache-defeating constants) |
| `dispatch-batch` | 1000 empty dispatches in one submission, wall time to completion |
| `dispatch-submit` | 100 one-dispatch submissions, each waited on |
| `alu` | 4M invocations × 256 hash rounds, 8 dispatches per submission |
| `stream` | 16M-word buffers, a*3+b, 8 dispatches per submission (~GB/s) |
| `roundtrip` | 16 MiB CPU write → copy → CPU read |

Each case runs 3 warm-up and 15 measured samples; median and min are reported. Samples are
wall-clock from submit to completion, so they include each runtime's validation and scheduling cost.

## Run

```sh
# SAGA (from the repository root)
swift run -c release SAGABenchmark

# wgpu (needs a Rust toolchain; builds the sibling ../wgpu checkout read-only)
cargo run --release --manifest-path Benchmarks/wgpu-bench/Cargo.toml
```

SAGA also takes `--timer-ms <n>` (Windows timer resolution) and `--shaders <dir>`.
Arguments for both: `--only <case>` (repeatable), `--samples <n>`. wgpu also honours
`WGPU_BACKEND=vulkan|dx12` (default `vulkan`).

Run `swift run -c release SAGABenchmark --only dispatch-replay --samples 100` to measure
the first submission of a compact compute list and its second submission after completion.
The benchmark records 1,000 dispatches per list outside the timed interval. It runs both an
empty shader and a shader that increments a storage-buffer counter, then verifies the counter.

The Vulkan backend can retain one completed compact list of direct compute dispatches.
Resubmit the same `CommandList` value to reuse its native recording. The backend discards
the recording when a different eligible list takes its place, when a referenced buffer or
pipeline is destroyed, or when the device closes. If the recording is still in flight, the
backend records another submission separately.
