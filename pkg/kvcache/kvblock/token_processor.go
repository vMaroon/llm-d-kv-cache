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

package kvblock

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/fxamacker/cbor/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
)

// defaultBlockSize is the default number of tokens per block.
// 16 is the default value used by vLLM.
const defaultBlockSize = 16

// noneHashLen is the length in bytes of vLLM's NONE_HASH and of every chained
// parent digest used during block hashing.
const noneHashLen = sha256.Size

// BlockHashBytes is the full 32-byte SHA-256 digest of a block, matching the
// raw form vLLM publishes when VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 and the
// chain input it uses internally regardless of the publish mode.
type BlockHashBytes [noneHashLen]byte

// EmptyBlockHashBytes is the all-zero parent that signals "start the chain
// from NONE_HASH" (i.e. tokens[0:block_size] is the first block of the
// request). Use this when you have the full prompt and want to reproduce vLLM
// from scratch; pass any other value to resume from a known prior digest.
var EmptyBlockHashBytes = BlockHashBytes{}

// IsEmpty reports whether b is the zero value (i.e. "start from NONE_HASH").
func (b BlockHashBytes) IsEmpty() bool { return b == EmptyBlockHashBytes }

// Truncate maps a 32-byte digest to vLLM's 64-bit publish form
// (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1):
//
//	uint64 = int.from_bytes(digest, "big") & ((1 << 64) - 1)
//
// equivalently the big-endian uint64 read from digest[24:32].
func (b BlockHashBytes) Truncate() BlockHash {
	return BlockHash(binary.BigEndian.Uint64(b[24:32]))
}

// TokenProcessorConfig holds the configuration for the token processor.
type TokenProcessorConfig struct {
	BlockSize int `json:"blockSize"`
	// HashSeed mirrors vLLM's PYTHONHASHSEED. It must match the value used by
	// the vLLM workers whose events you want to reproduce. With sha256_cbor,
	// vLLM computes:
	//
	//	NONE_HASH = sha256(cbor.canonical(<seed-string>))
	//
	// so a string is hashed (not an integer); empty string is allowed.
	HashSeed string `json:"hashSeed"`
}

// DefaultTokenProcessorConfig returns the default configuration for the token processor.
func DefaultTokenProcessorConfig() *TokenProcessorConfig {
	return &TokenProcessorConfig{
		BlockSize: defaultBlockSize,
		HashSeed:  "",
	}
}

// TokenProcessor defines the interface for converting tokens to KVBlockKeys.
//
// Implementations must produce hashes that exactly match what vLLM publishes
// in BlockStored events when configured with prefix_caching_hash_algo =
// "sha256_cbor". Both publish modes are supported:
//
//   - VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 (default): events carry the
//     truncated 64-bit form. Use TokensToKVBlockKeys, which returns BlockHash
//     (uint64). Reproducing vLLM exactly requires parentKey=EmptyBlockHash and
//     the full prompt — the truncated form has no inverse, so chained calls
//     from a non-empty 64-bit parent are locally consistent but cannot match
//     vLLM.
//
//   - VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0: events carry the full 32-byte
//     SHA-256 digest. Use TokensToKVBlockHashBytes, which accepts a 32-byte
//     parent and returns 32-byte digests. Chained calls from any prior digest
//     reproduce vLLM exactly.
type TokenProcessor interface {
	// TokensToKVBlockKeys converts tokens into truncated 64-bit block keys.
	//
	// parentKey is the truncated 64-bit hash of the block immediately
	// preceding the first chunk in tokens, or EmptyBlockHash to start a fresh
	// chain (i.e., the prompt begins at block 0). To reproduce vLLM-emitted
	// hashes exactly, pass the full prompt with EmptyBlockHash.
	//
	// modelName is accepted for API compatibility and ignored: vLLM does not
	// mix the model name into block hashes.
	//
	// extraFeatures supplies per-block extras (LoRA name, multi-modal item
	// identifiers, cache-salt, prompt-embeds hash). nil means text-only with
	// no extras for any block. When non-nil, length must equal the number of
	// full token chunks; nil entries inside mean "no extras" for that chunk.
	TokensToKVBlockKeys(
		parentKey BlockHash, tokens []uint32, modelName string,
		extraFeatures []*BlockExtraFeatures,
	) ([]BlockHash, error)

	// TokensToKVBlockHashBytes converts tokens into full 32-byte block
	// digests, matching vLLM's bytes publish mode.
	//
	// parent is the 32-byte digest of the block preceding the first chunk, or
	// EmptyBlockHashBytes to start from NONE_HASH. Unlike the uint64 path,
	// chained calls from a non-empty parent reproduce vLLM exactly because
	// the digest IS the chain input.
	//
	// Other arguments behave identically to TokensToKVBlockKeys.
	TokensToKVBlockHashBytes(
		parent BlockHashBytes, tokens []uint32, modelName string,
		extraFeatures []*BlockExtraFeatures,
	) ([]BlockHashBytes, error)

	// BlockSize returns the number of tokens per block used by this processor.
	BlockSize() int
}

// chunkedTokenDatabase is a vLLM-parity implementation of TokenProcessor that
// uses sha256_cbor over the same (parent, tokens, extras) triple vLLM hashes,
// then truncates each digest to 64 bits via the same mapping vLLM uses when
// VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1, namely:
//
//	uint64 = int.from_bytes(digest, "big") & ((1 << 64) - 1)
//
// which is the big-endian uint64 read from digest[24:32].
type chunkedTokenDatabase struct {
	TokenProcessorConfig
	encoder      cbor.EncMode
	noneHashFull [noneHashLen]byte
}

var _ TokenProcessor = &chunkedTokenDatabase{}

// NewChunkedTokenDatabase creates a new TokenProcessor.
func NewChunkedTokenDatabase(config *TokenProcessorConfig) (TokenProcessor, error) {
	if config == nil {
		config = DefaultTokenProcessorConfig()
	}

	if config.BlockSize <= 0 {
		return nil, fmt.Errorf("blockSize must be greater than 0, got %d", config.BlockSize)
	}

	encoder, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("failed to create CBOR encoder: %w", err)
	}

	// vLLM init_none_hash: NONE_HASH = sha256_cbor(<seed-string>).
	// The seed is always treated as a string (PYTHONHASHSEED is read with
	// os.getenv, which returns str), even when it looks numeric.
	seedCBOR, err := encoder.Marshal(config.HashSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to CBOR-encode hash seed: %w", err)
	}
	noneHash := sha256.Sum256(seedCBOR)

	return &chunkedTokenDatabase{
		TokenProcessorConfig: *config,
		encoder:              encoder,
		noneHashFull:         noneHash,
	}, nil
}

// hashBlock computes one block's full 32-byte SHA-256 digest over the
// canonical CBOR encoding of (parent, tokens, extra), matching
// vllm.utils.hashing.sha256_cbor((parent, tuple(tokens), extra)).
//
// extra must already be vLLM-shaped: a flat []any of strings/[]byte/ints, or
// nil for "no extras". A non-nil but empty []any is NOT equivalent to nil;
// callers must pass nil when there are no extras.
func (db *chunkedTokenDatabase) hashBlock(parent []byte, tokens []uint32, extra any) (BlockHashBytes, error) {
	payload := []any{parent, tokens, extra}
	b, err := db.encoder.Marshal(payload)
	if err != nil {
		return BlockHashBytes{}, fmt.Errorf("CBOR marshal failed: %w", err)
	}
	return sha256.Sum256(b), nil
}

// expandUint64ToParent maps a truncated 64-bit parent key back to the 32-byte
// chain input. There is no general inverse (only 8 of 32 bytes are known), so
// this is a deterministic local extension and does NOT match vLLM's chain.
//
// We zero-pad the high 24 bytes and place the uint64 in the low 8 bytes
// (big-endian). Used only when callers chain from a non-Empty parentKey on
// the truncated uint64 path.
func expandUint64ToParent(v uint64) BlockHashBytes {
	var buf BlockHashBytes
	binary.BigEndian.PutUint64(buf[24:32], v)
	return buf
}

// chunkTokens splits the input slice of tokens into chunks of size blockSize.
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	bs := db.TokenProcessorConfig.BlockSize
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += bs {
		end := i + bs
		if end > len(tokens) {
			break // no partial blocks
		}
		chunks = append(chunks, tokens[i:end])
	}
	return chunks
}

// BlockSize returns the number of tokens per block.
func (db *chunkedTokenDatabase) BlockSize() int {
	return db.TokenProcessorConfig.BlockSize
}

// hashChain is the shared primitive behind both publish modes. It produces
// the full 32-byte digest per block, chaining the digest as the parent for
// the next block exactly as vLLM does. Caller picks the starting parent.
func (db *chunkedTokenDatabase) hashChain(
	parent BlockHashBytes, tokens []uint32,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHashBytes, error) {
	chunks := db.chunkTokens(tokens)
	if len(chunks) == 0 {
		return nil, nil
	}

	if extraFeatures == nil {
		extraFeatures = make([]*BlockExtraFeatures, len(chunks))
	} else if len(extraFeatures) != len(chunks) {
		return nil, fmt.Errorf(
			"extraFeatures length %d does not match token chunk count %d (blockSize=%d, tokens=%d)",
			len(extraFeatures), len(chunks),
			db.TokenProcessorConfig.BlockSize, len(tokens))
	}

	logger := log.FromContext(context.Background())

	out := make([]BlockHashBytes, len(chunks))
	cur := parent
	for i, chunk := range chunks {
		extra := extraFeatures[i].cborExtras()
		digest, err := db.hashBlock(cur[:], chunk, extra)
		if err != nil {
			logger.Error(err, "failed to hash block", "blockIdx", i)
			return nil, err
		}
		cur = digest
		out[i] = digest
	}
	return out, nil
}

// TokensToKVBlockHashBytes is the bytes-mode entry point. parent may be
// EmptyBlockHashBytes to start from NONE_HASH, or any prior digest to resume.
func (db *chunkedTokenDatabase) TokensToKVBlockHashBytes(
	parent BlockHashBytes, tokens []uint32, _ string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHashBytes, error) {
	if parent.IsEmpty() {
		parent = db.noneHashFull
	}
	return db.hashChain(parent, tokens, extraFeatures)
}

// TokensToKVBlockKeys delegates to the bytes path and truncates each digest
// to vLLM's 64-bit publish form. Chained computation from a non-Empty
// parentKey uses a deterministic but non-vLLM-matching expansion of the
// uint64 (see expandUint64ToParent); pass EmptyBlockHash + the full prompt
// for true vLLM parity.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(
	parentKey BlockHash, tokens []uint32, _ string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
	var parent BlockHashBytes
	if parentKey == EmptyBlockHash {
		parent = db.noneHashFull
	} else {
		parent = expandUint64ToParent(uint64(parentKey))
	}

	digests, err := db.hashChain(parent, tokens, extraFeatures)
	if err != nil {
		return nil, err
	}
	return utils.SliceMap(digests, func(d BlockHashBytes) BlockHash {
		return d.Truncate()
	}), nil
}
