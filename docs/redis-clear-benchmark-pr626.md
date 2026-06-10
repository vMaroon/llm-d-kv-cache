# Redis Clear Reverse-Index Benchmark

This document records the local benchmark data used to evaluate the Redis
`Clear` change in the PR branch. The benchmarks were run against `miniredis`,
so they measure Go client/protocol/Lua behavior without real network jitter or
Redis server deployment effects.

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
| 512 | 512 | 22.15 ms/op | 4.52 ms/op | 4.91x |
| 2,048 | 2,048 | 105.16 ms/op | 16.21 ms/op | 6.49x |
| 8,192 | 8,192 | 440.37 ms/op | 57.72 ms/op | 7.63x |
| 32,768 | 32,768 | 2,498.61 ms/op | 321.95 ms/op | 7.76x |

Raw output:

```text
BenchmarkRedisClearScaleLegacyFullScan/keys=512/unrelated=512/pods=8-24             3      22152971 ns/op       1352733 B/op      58066 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=2048/unrelated=2048/pods=8-24           3     105158498 ns/op       6290752 B/op     231910 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=8192/unrelated=8192/pods=8-24           3     440367440 ns/op      43528690 B/op     927371 allocs/op
BenchmarkRedisClearScaleLegacyFullScan/keys=32768/unrelated=32768/pods=8-24         3    2498608155 ns/op     503204013 B/op    3709468 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=512/unrelated=512/pods=8-24               3       4516145 ns/op      10632394 B/op      32265 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=2048/unrelated=2048/pods=8-24             3      16206379 ns/op      42467901 B/op     127358 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=8192/unrelated=8192/pods=8-24             3      57717958 ns/op     170702813 B/op     507548 allocs/op
BenchmarkRedisClearScaleReverseIndex/keys=32768/unrelated=32768/pods=8-24           3     321952799 ns/op     696382456 B/op    2028268 allocs/op
```

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
| Clear | 1,024 keys, 8 pods | 8.11 ms/op |

Raw output:

```text
BenchmarkRedisIndexOperations/Add/keys=1024/pods=1-24             100      10435609 ns/op       3670311 B/op      76639 allocs/op
BenchmarkRedisIndexOperations/Lookup/keys=1024/pods=1-24          100       3065127 ns/op        733476 B/op      23489 allocs/op
BenchmarkRedisIndexOperations/GetRequestKey-24                    100         12466 ns/op           667 B/op         30 allocs/op
BenchmarkRedisIndexOperations/EvictRequestKey/pods=1-24           100        115220 ns/op        195404 B/op        784 allocs/op
BenchmarkRedisIndexOperations/EvictEngineKey/pods=1-24            100        253226 ns/op        390092 B/op       1556 allocs/op
BenchmarkRedisIndexOperations/Clear/keys=1024/pods=8-24           100       8113959 ns/op      21217115 B/op      63835 allocs/op
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
scales with the pod-owned reverse index instead of the entire Redis keyspace. In
the local Clear scaling benchmark, the speedup grows from about 4.9x at 512
request keys to about 7.8x at 32k request keys with an equal number of unrelated
Redis hash keys.
