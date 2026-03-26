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

// dump_zmq_payload connects to a vLLM ZMQ publisher and saves raw msgpack
// payloads to disk for replay with decode_msgpack_payload.  Adjust the
// variables below and run with:
//
//	go run ./examples/zmq_kv_events/dump_zmq_payload/
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
)

func main() {
	endpoint := "tcp://localhost:5557"
	outFile := "examples/testdata/block_stored_example.msgpack"

	logger := slog.Default()

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

	msg, err := sub.Recv()
	if err != nil {
		panic(err)
	}

	parts := msg.Frames
	if len(parts) != 3 {
		panic(fmt.Sprintf("unexpected frame count: got %d, want 3", len(parts)))
	}

	topic := string(parts[0])
	seq := binary.BigEndian.Uint64(parts[1])
	payload := parts[2]

	if err := os.WriteFile(outFile, payload, 0600); err != nil {
		panic(err)
	}

	logger.Info("saved payload", "file", outFile, "topic", topic, "seq", seq, "bytes", len(payload))
}
