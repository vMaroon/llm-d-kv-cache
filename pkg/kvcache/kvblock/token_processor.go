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
// "sha256_cbor" and the default VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1.
type TokenProcessor interface {
	// TokensToKVBlockKeys converts tokens into block keys.
	//
	// parentKey is the truncated 64-bit hash of the block immediately preceding
	// the first chunk in tokens, or EmptyBlockHash to start a fresh chain
	// (i.e., the prompt begins at block 0). To reproduce vLLM-emitted hashes
	// exactly, callers must pass the full prompt with EmptyBlockHash; chained
	// computation from a non-Empty parentKey is locally consistent but does
	// not match vLLM, because the chain input there is the full 32-byte SHA-256
	// digest of the parent block, not the truncated 64-bit form.
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
func (db *chunkedTokenDatabase) hashBlock(parent []byte, tokens []uint32, extra any) ([noneHashLen]byte, error) {
	payload := []any{parent, tokens, extra}
	b, err := db.encoder.Marshal(payload)
	if err != nil {
		return [noneHashLen]byte{}, fmt.Errorf("CBOR marshal failed: %w", err)
	}
	return sha256.Sum256(b), nil
}

// truncateDigest mirrors vLLM's maybe_convert_block_hash with
// VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1: low 64 bits of the big-endian
// integer interpretation of the 32-byte digest, i.e. digest[24:32] read as
// big-endian uint64.
func truncateDigest(d [noneHashLen]byte) uint64 {
	return binary.BigEndian.Uint64(d[24:32])
}

// expandUint64ToParent maps a truncated 64-bit parent key back to the 32-byte
// chain input. There is no general inverse (only 8 of 32 bytes are known), so
// this is a deterministic local extension and does NOT match vLLM's chain.
//
// We zero-pad the high 24 bytes and place the uint64 in the low 8 bytes
// (big-endian). Used only when callers chain from a non-Empty parentKey.
func expandUint64ToParent(v uint64) [noneHashLen]byte {
	var buf [noneHashLen]byte
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

// TokensToKVBlockKeys converts tokens into block keys.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(
	parentKey BlockHash, tokens []uint32, _ string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
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

	var parent [noneHashLen]byte
	if parentKey == EmptyBlockHash {
		parent = db.noneHashFull
	} else {
		parent = expandUint64ToParent(uint64(parentKey))
	}

	logger := log.FromContext(context.Background())

	out := make([]BlockHash, len(chunks))
	for i, chunk := range chunks {
		extra := extraFeatures[i].cborExtras()
		digest, err := db.hashBlock(parent[:], chunk, extra)
		if err != nil {
			logger.Error(err, "failed to hash block", "blockIdx", i)
			return nil, err
		}
		parent = digest
		out[i] = BlockHash(truncateDigest(digest))
	}

	return utils.SliceMap(out, func(h BlockHash) BlockHash { return h }), nil
}
