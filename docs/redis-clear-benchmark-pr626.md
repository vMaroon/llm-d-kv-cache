# Redis Clear Reverse-Index Benchmark

This document records the local benchmark data used to evaluate the Redis
`Clear` change in the PR branch. The benchmarks were run against `miniredis`,
so they measure Go client/protocol behavior without real network jitter or Redis
server deployment effects.

## Environment

- Date: 2026-06-10
- OS/arch: linux/amd64
- CPU: Intel(R) Core(TM) Ultra 7 270K Plus
- Package: `github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock`

## Clear Scaling

The clear scaling benchmark compares the legacy global keyspace scan with the
reverse-index clear path. Each case uses 8 pods and the same number of unrelated
Redis hash keys as request keys.

Command:

```bash
go test ./pkg/kvcache/kvblock \
  -run '^$' \
  -bench 'BenchmarkRedisClearScale' \
  -benchmem \
  -benchtime=3x \
  -count=1
```

| request keys | unrelated keys | legacy full scan | reverse index | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 20.11 ms/op | 5.41 ms/op | 3.72x |
| 2,048 | 2,048 | 106.35 ms/op | 12.70 ms/op | 8.37x |
| 8,192 | 8,192 | 457.68 ms/op | 54.57 ms/op | 8.39x |
| 32,768 | 32,768 | 2,533.05 ms/op | 329.26 ms/op | 7.69x |

Raw output:

```text
BenchmarkRedisClearScaleLegacyFullScan/keys=512/unrelated=512/pods=8-24             3      20112054 ns/op       1353368 B/op      58068 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=2048/unrelated=2048/pods=8-24           3     106345535 ns/op       6290458 B/op     231911 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=8192/unrelated=8192/pods=8-24           3     457683679 ns/op      43526186 B/op     927366 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=32768/unrelated=32768/pods=8-24         3    2533046849 ns/op     503208552 B/op    3709527 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=512/unrelated=512/pods=8-24               3       5410594 ns/op        590341 B/op      20169 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=2048/unrelated=2048/pods=8-24             3      12698848 ns/op       2361600 B/op      80178 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=8192/unrelated=8192/pods=8-24             3      54571458 ns/op      10408296 B/op     320176 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=32768/unrelated=32768/pods=8-24           3     329264781 ns/op      55362544 B/op    1280123 allocs/op
```

## Lua Variant Check

Before settling on the current no-Lua cleanup, I also tested a Clear-specific
Lua variant that batched request hashes and reverse-index members into a script.
The no-Lua implementation was faster in this `miniredis` benchmark while keeping
the production code smaller. It also avoids Redis Cluster multi-key Lua script
constraints for the Clear cleanup path.

| request keys | Lua reverse index | no-Lua reverse index | no-Lua comparison |
| ---: | ---: | ---: | ---: |
| 512 | 4.52 ms/op | 2.75 ms/op | 1.64x faster |
| 2,048 | 16.21 ms/op | 12.41 ms/op | 1.31x faster |
| 8,192 | 57.72 ms/op | 55.61 ms/op | 1.04x faster |
| 32,768 | 321.95 ms/op | 306.18 ms/op | 1.05x faster |

The current implementation therefore keeps `Clear` as a pipelined `HDEL` +
`SREM` cleanup over the pod-owned reverse-index set, instead of adding a
Clear-specific Lua script.

## Redis Index Operations

The operation benchmark measures the current Redis index implementation after
the reverse-index change.

Command:

```bash
go test ./pkg/kvcache/kvblock \
  -run '^$' \
  -bench 'BenchmarkRedisIndexOperations' \
  -benchmem \
  -benchtime=100x \
  -count=1
```

| operation | workload | result |
| --- | ---: | ---: |
| Add | 1,024 keys, 1 pod | 10.44 ms/op |
| Lookup | 1,024 keys, 1 pod | 3.07 ms/op |
| GetRequestKey | 1 key | 12.47 us/op |
| Evict(RequestKey) | 1 key, 1 pod | 115.22 us/op |
| Evict(EngineKey) | 1 key, 1 pod | 253.23 us/op |
| Clear | 1,024 keys, 8 pods | 6.59 ms/op |

Raw output:

```text
BenchmarkRedisIndexOperations/Add/keys=1024/pods=1-24             100      10435609 ns/op       3670311 B/op      76639 allocs/op
BenchmarkRedisIndexOperations/Lookup/keys=1024/pods=1-24          100       3065127 ns/op        733476 B/op      23489 allocs/op
BenchmarkRedisIndexOperations/GetRequestKey-24                    100         12466 ns/op           667 B/op         30 allocs/op
BenchmarkRedisIndexOperations/EvictRequestKey/pods=1-24           100        115220 ns/op        195404 B/op        784 allocs/op
BenchmarkRedisIndexOperations/EvictEngineKey/pods=1-24            100        253226 ns/op        390092 B/op       1556 allocs/op
BenchmarkRedisIndexOperations/Clear/keys=1024/pods=8-24           100       6585618 ns/op       1146185 B/op      40037 allocs/op
```

## Existing Profiling Benchmarks

The repository already has Redis Add/Lookup profiling benchmarks under
`tests/profiling/kv_cache_index`. Those benchmarks use 10,000 keys.

Command:

```bash
go test ./tests/profiling/kv_cache_index \
  -run '^$' \
  -bench 'BenchmarkRedis' \
  -benchmem \
  -benchtime=100x \
  -count=1
```

Raw output:

```text
BenchmarkRedis_Add-24        100      58031984 ns/op      23117263 B/op     710444 allocs/op
BenchmarkRedis_Lookup-24     100      29830566 ns/op       7578859 B/op     230123 allocs/op
```

## Takeaway

The reverse-index clear path avoids the legacy global `SCAN "*"` behavior and
scales with the pod-owned reverse index instead of the entire Redis keyspace. It
does not require a Clear-specific Lua script; the cleanup is pipelined with
`HDEL` and `SREM`. In the local Clear scaling benchmark, the speedup ranges from
about 3.7x to 8.4x with an equal number of unrelated Redis hash keys.
