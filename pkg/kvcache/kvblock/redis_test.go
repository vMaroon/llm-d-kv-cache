/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvblock_test

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-logr/logr"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// createRedisIndexForTesting creates a new RedisIndex with a mock Redis server for testing.
func createRedisIndexForTesting(t *testing.T) Index {
	t.Helper()
	index, _ := createRedisIndexAndServerForTesting(t)
	return index
}

func createRedisIndexAndServerForTesting(t *testing.T) (*RedisIndex, *miniredis.Miniredis) {
	t.Helper()
	server, err := miniredis.Run()
	require.NoError(t, err)

	t.Cleanup(func() {
		server.Close()
	})

	redisConfig := &RedisIndexConfig{
		Address: server.Addr(),
	}
	index, err := NewRedisIndex(redisConfig)
	require.NoError(t, err)
	redisIndex, ok := index.(*RedisIndex)
	require.True(t, ok)
	return redisIndex, server
}

// TestRedisIndexBehavior tests the Redis index implementation using common test behaviors.
func TestRedisIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createRedisIndexForTesting)
}

func TestRedisClearPreservesUnrelatedKeys(t *testing.T) {
	index, _ := createRedisIndexAndServerForTesting(t)
	ctx := t.Context()
	pod := PodEntry{PodIdentifier: "pod-clear", DeviceTier: "gpu"}
	requestKey := BlockHash(0xC1EA1001)

	require.NoError(t, index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod}))
	require.NoError(t, index.RedisClient.HSet(ctx, "unrelated-hash", pod.String(), "keep").Err())
	require.NoError(t, index.RedisClient.Set(ctx, "unrelated-string", "keep", 0).Err())

	require.NoError(t, index.Clear(ctx, pod.PodIdentifier))

	gotHash, err := index.RedisClient.HGet(ctx, "unrelated-hash", pod.String()).Result()
	require.NoError(t, err)
	assert.Equal(t, "keep", gotHash)

	gotString, err := index.RedisClient.Get(ctx, "unrelated-string").Result()
	require.NoError(t, err)
	assert.Equal(t, "keep", gotString)

	hits, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, hits[requestKey])
}

func TestRedisEvictPrunesReverseIndex(t *testing.T) {
	index, _ := createRedisIndexAndServerForTesting(t)
	ctx := t.Context()
	pod := PodEntry{PodIdentifier: "pod-evict", DeviceTier: "gpu"}
	requestKey := BlockHash(0xC1EA1002)
	member := fmt.Sprintf("%s\x00%s", requestKey.String(), pod.String())
	podEntriesKey := "kvblock:pod:" + pod.PodIdentifier + ":entries"

	require.NoError(t, index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod}))
	ok, err := index.RedisClient.SIsMember(ctx, podEntriesKey, member).Result()
	require.NoError(t, err)
	require.True(t, ok)

	require.NoError(t, index.Evict(ctx, requestKey, RequestKey, []PodEntry{pod}))

	ok, err = index.RedisClient.SIsMember(ctx, podEntriesKey, member).Result()
	require.NoError(t, err)
	assert.False(t, ok)
}

func TestRedisClearDropsMalformedReverseEntries(t *testing.T) {
	index, _ := createRedisIndexAndServerForTesting(t)
	ctx := t.Context()
	podEntriesKey := "kvblock:pod:pod-malformed:entries"

	require.NoError(t, index.RedisClient.SAdd(ctx, podEntriesKey, "malformed").Err())

	require.NoError(t, index.Clear(ctx, "pod-malformed"))

	count, err := index.RedisClient.SCard(ctx, podEntriesKey).Result()
	require.NoError(t, err)
	assert.Zero(t, count)
}

func BenchmarkRedisClearLegacyFullScan(b *testing.B) {
	benchmarkRedisClear(b, false)
}

func BenchmarkRedisClearReverseIndex(b *testing.B) {
	benchmarkRedisClear(b, true)
}

func benchmarkRedisClear(b *testing.B, reverseIndex bool) {
	server, err := miniredis.Run()
	require.NoError(b, err)
	b.Cleanup(server.Close)

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(b, err)
	redisIndex := index.(*RedisIndex)
	ctx := log.IntoContext(context.Background(), logr.Discard())
	const targetPod = "pod-0"

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		require.NoError(b, redisIndex.RedisClient.FlushDB(ctx).Err())
		seedRedisClearBenchmarkData(b, ctx, redisIndex, reverseIndex)
		b.StartTimer()

		if reverseIndex {
			require.NoError(b, redisIndex.Clear(ctx, targetPod))
		} else {
			require.NoError(b, legacyRedisClear(ctx, redisIndex.RedisClient, targetPod))
		}
	}
}

func seedRedisClearBenchmarkData(b *testing.B, ctx context.Context, index *RedisIndex, reverseIndex bool) {
	b.Helper()
	const (
		keyCount       = 512
		podCount       = 8
		unrelatedCount = 512
	)

	entries := make([]PodEntry, 0, podCount)
	for pod := 0; pod < podCount; pod++ {
		entries = append(entries, PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", pod), DeviceTier: "gpu"})
	}

	if reverseIndex {
		for key := 0; key < keyCount; key++ {
			blockHash := BlockHash(key + 1)
			require.NoError(b, index.Add(ctx, nil, []BlockHash{blockHash}, entries))
		}
	} else {
		pipe := index.RedisClient.Pipeline()
		for key := 0; key < keyCount; key++ {
			redisKey := strconv.Itoa(key + 1)
			for _, entry := range entries {
				pipe.HSet(ctx, redisKey, entry.String(), "")
			}
		}
		_, err := pipe.Exec(ctx)
		require.NoError(b, err)
	}

	pipe := index.RedisClient.Pipeline()
	for key := 0; key < unrelatedCount; key++ {
		pipe.HSet(ctx, fmt.Sprintf("unrelated:%d", key), "other@gpu", "")
	}
	_, err := pipe.Exec(ctx)
	require.NoError(b, err)
}

func legacyRedisClear(ctx context.Context, client *redis.Client, podIdentifier string) error {
	fieldPrefix := podIdentifier + "@"
	const scanBatch int64 = 1024
	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, "*", scanBatch).Result()
		if err != nil {
			return err
		}
		for _, key := range keys {
			if key == "" || len(key) >= len("engine:") && key[:len("engine:")] == "engine:" {
				continue
			}
			fields, err := client.HKeys(ctx, key).Result()
			if err != nil {
				return err
			}
			var stale []string
			for _, field := range fields {
				if len(field) >= len(fieldPrefix) && field[:len(fieldPrefix)] == fieldPrefix {
					stale = append(stale, field)
				}
			}
			if len(stale) == 0 {
				continue
			}
			if err := client.HDel(ctx, key, stale...).Err(); err != nil {
				return err
			}
			if err := pruneEmptyHashForBenchmark(ctx, client, key); err != nil {
				return err
			}
		}
		if cursor = next; cursor == 0 {
			return nil
		}
	}
}

func pruneEmptyHashForBenchmark(ctx context.Context, client *redis.Client, key string) error {
	n, err := client.HLen(ctx, key).Result()
	if err != nil {
		return err
	}
	if n == 0 {
		return client.Del(ctx, key).Err()
	}
	return nil
}
