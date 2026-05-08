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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// vllmFixture mirrors the JSON layout produced by
// pkg/kvcache/kvblock/testdata/gen_fixtures.py.
type vllmFixture struct {
	Name        string   `json:"name"`
	Seed        string   `json:"seed"`
	BlockSize   int      `json:"block_size"`
	Tokens      []uint32 `json:"tokens"`
	Extras      [][]any  `json:"extras"`
	DigestsHex  []string `json:"digests_hex"`
	Truncated   []uint64 `json:"truncated_uint64"`
	NoneHashHex string   `json:"none_hash_hex"`
}

func loadVLLMFixtures(t *testing.T) []vllmFixture {
	t.Helper()
	path := filepath.Join("testdata", "vllm_block_hashes.json")
	data, err := os.ReadFile(path)
	require.NoError(t, err, "read %s", path)
	var fixtures []vllmFixture
	require.NoError(t, json.Unmarshal(data, &fixtures), "parse %s", path)
	require.NotEmpty(t, fixtures)
	return fixtures
}

// TestVLLMParity_NoneHash verifies that the encoder we use for canonical CBOR
// reproduces vLLM's NONE_HASH = sha256(cbor.canonical(<seed-string>)).
func TestVLLMParity_NoneHash(t *testing.T) {
	enc, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	for _, fx := range loadVLLMFixtures(t) {
		t.Run(fx.Name, func(t *testing.T) {
			seedCBOR, err := enc.Marshal(fx.Seed)
			require.NoError(t, err)
			got := sha256.Sum256(seedCBOR)
			require.Equal(t, fx.NoneHashHex, hex.EncodeToString(got[:]),
				"NONE_HASH mismatch for seed %q", fx.Seed)
		})
	}
}

// TestVLLMParity_BlockHashes runs every fixture through the production
// TokenProcessor and asserts the truncated 64-bit hashes match the values
// vLLM would publish in BlockStored events with VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1.
//
// extras in the JSON are decoded as [][]any (one element list per block); we
// pass them through ParseRawExtraKeys so we exercise the wire-decoding path.
func TestVLLMParity_BlockHashes(t *testing.T) {
	for _, fx := range loadVLLMFixtures(t) {
		t.Run(fx.Name, func(t *testing.T) {
			proc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
				BlockSize: fx.BlockSize,
				HashSeed:  fx.Seed,
			})
			require.NoError(t, err)

			var extras []*kvblock.BlockExtraFeatures
			if fx.Extras != nil {
				extras, err = kvblock.ParseRawExtraKeys(fx.Extras)
				require.NoError(t, err)
			}

			got, err := proc.TokensToKVBlockKeys(
				kvblock.EmptyBlockHash, fx.Tokens, "ignored-model-name", extras,
			)
			require.NoError(t, err)
			require.Len(t, got, len(fx.Truncated),
				"block count mismatch for %s", fx.Name)

			for i, want := range fx.Truncated {
				require.Equalf(t, want, uint64(got[i]),
					"%s block %d: got %d, want %d", fx.Name, i, uint64(got[i]), want)
			}
		})
	}
}

// TestVLLMParity_StructuredExtras exercises the structured-extras path
// (LoraName / MMHashes / CacheSalt) instead of RawExtras, to ensure the
// structured shape produces the same CBOR as vLLM's flat tuple.
func TestVLLMParity_StructuredExtras(t *testing.T) {
	// Re-use lora_only / cache_salt_first_block_only / lora_mm_salt fixtures
	// but rebuild the extras through the structured fields.
	type extraBuilder func() []*kvblock.BlockExtraFeatures
	cases := map[string]extraBuilder{
		"lora_only": func() []*kvblock.BlockExtraFeatures {
			b := []*kvblock.BlockExtraFeatures{
				{LoraName: "my-lora"},
				{LoraName: "my-lora"},
			}
			return b
		},
		"cache_salt_first_block_only": func() []*kvblock.BlockExtraFeatures {
			return []*kvblock.BlockExtraFeatures{
				{CacheSalt: "salt-xyz"},
				nil,
			}
		},
		"lora_mm_salt": func() []*kvblock.BlockExtraFeatures {
			return []*kvblock.BlockExtraFeatures{
				{
					LoraName: "my-lora",
					MMHashes: []kvblock.MMHash{
						{Hash: "mm_hash_a"},
						{Hash: "mm_hash_b"},
					},
					CacheSalt: "salt-xyz",
				},
				{
					LoraName: "my-lora",
					MMHashes: []kvblock.MMHash{{Hash: "mm_hash_b"}},
				},
			}
		},
	}

	fixtures := loadVLLMFixtures(t)
	byName := make(map[string]vllmFixture, len(fixtures))
	for _, fx := range fixtures {
		byName[fx.Name] = fx
	}

	for name, build := range cases {
		fx, ok := byName[name]
		require.Truef(t, ok, "missing fixture %q", name)

		t.Run(name, func(t *testing.T) {
			proc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
				BlockSize: fx.BlockSize,
				HashSeed:  fx.Seed,
			})
			require.NoError(t, err)

			got, err := proc.TokensToKVBlockKeys(
				kvblock.EmptyBlockHash, fx.Tokens, "model", build(),
			)
			require.NoError(t, err)
			require.Len(t, got, len(fx.Truncated))

			for i, want := range fx.Truncated {
				require.Equalf(t, want, uint64(got[i]),
					"%s structured block %d: got %d, want %d",
					name, i, uint64(got[i]), want)
			}
		})
	}
}

// TestVLLMParity_NilVsEmptyExtras documents the asymmetry between "no extras
// for this block" (CBOR null) and "empty extras list" (CBOR []), which would
// produce different hashes if confused. Our cborExtras() helper folds both
// nil receivers and empty fields/RawExtras into nil → CBOR null, matching vLLM.
func TestVLLMParity_NilVsEmptyExtras(t *testing.T) {
	proc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "0",
	})
	require.NoError(t, err)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	// Reference value: extras=nil — same as fixture "plain_one_block".
	wantBlocks, err := proc.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "model", nil,
	)
	require.NoError(t, err)
	require.Len(t, wantBlocks, 1)

	// Equivalent forms that must produce the same hash:
	equivalents := [][]*kvblock.BlockExtraFeatures{
		{nil},
		{{}},
		{{RawExtras: []any{}}},
		{{MMHashes: nil}},
	}
	for i, ef := range equivalents {
		got, err := proc.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", ef)
		require.NoErrorf(t, err, "equivalent #%d", i)
		require.Equalf(t, wantBlocks[0], got[0], "equivalent #%d should match nil-extras", i)
	}
}
