// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// zmq_consumer connects to a vLLM ZMQ publisher, parses KV-cache events, and
// maintains an in-memory KV block index.  Adjust the variables below and run
// with:
//
//	go run ./examples/zmq_kv_events/zmq_consumer/
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

func main() {
	endpoint := "tcp://localhost:5557"

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	sub := zmq4.NewSub(ctx)
	defer sub.Close()

	if err := sub.Dial(endpoint); err != nil {
		panic(err)
	}
	logger.Info("connected", "endpoint", endpoint)

	if err := sub.SetOption(zmq4.OptionSubscribe, ""); err != nil {
		panic(err)
	}

	vllmAdapter := engineadapter.NewVLLMAdapter()

	index, err := kvblock.NewInMemoryIndex(nil)
	if err != nil {
		panic(err)
	}

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(nil)
	if err != nil {
		panic(err)
	}

	for {
		if ctx.Err() != nil {
			break
		}

		msg, err := sub.Recv()
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			panic(err)
		}

		parts := msg.Frames
		if len(parts) != 3 {
			logger.Warn("unexpected frame count, skipping", "got", len(parts))
			continue
		}

		topic := string(parts[0])
		seq := binary.BigEndian.Uint64(parts[1])
		payload := parts[2]

		logger.Info("received message", "topic", topic, "seq", seq, "payloadBytes", len(payload))

		podID, modelName, batch, err := vllmAdapter.ParseMessage(&kvevents.RawMessage{
			Topic:    topic,
			Sequence: seq,
			Payload:  payload,
		})
		if err != nil {
			logger.Error("failed to parse message", "err", err)
			continue
		}
		logger.Info("parsed message", "podID", podID, "modelName", modelName)

		for _, event := range batch.Events {
			switch ev := event.(type) {
			case *kvevents.BlockStoredEvent:
				podEntries := []kvblock.PodEntry{{PodIdentifier: podID, DeviceTier: "GPU"}}

				engineKeys := make([]kvblock.BlockHash, len(ev.BlockHashes))
				for i, h := range ev.BlockHashes {
					engineKeys[i] = kvblock.BlockHash(h)
				}

				parentRequestKey := kvblock.EmptyBlockHash
				if ev.ParentHash != 0 {
					key, err := index.GetRequestKey(ctx, kvblock.BlockHash(ev.ParentHash))
					if err != nil {
						logger.Warn("parent block not found, using empty key", "parentHash", ev.ParentHash)
					} else {
						parentRequestKey = key
					}
				}

				requestKeys, err := tokenProcessor.TokensToKVBlockKeys(parentRequestKey, ev.Tokens, "", ev.ExtraKeys)
				if err != nil {
					logger.Error("failed to compute request keys", "err", err)
					continue
				}
				logger.Info("mapped block keys", "engineKeys", engineKeys, "requestKeys", requestKeys)

				if len(engineKeys) > 0 {
					if err := index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
						logger.Error("failed to add block to index", "err", err)
					}
				}

			default:
				logger.Info("unhandled event type", "podID", podID, "type", fmt.Sprintf("%T", event))
			}
		}
	}

	logger.Info("shutting down")
}
