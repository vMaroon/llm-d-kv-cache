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

// decode_msgpack_payload reads a raw msgpack payload and pretty-prints the
// decoded KV-cache events.  Adjust the variables below and run with:
//
//	go run ./examples/zmq_kv_events/decode_msgpack_payload/
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

func main() {
	inputFile := "examples/testdata/block_stored_example.msgpack"
	topic := "kv@pod-1@test-model"

	payload, err := os.ReadFile(inputFile)
	if err != nil {
		panic(err)
	}

	adapter := engineadapter.NewVLLMAdapter()
	podID, modelName, batch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   topic,
		Payload: payload,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("podID: %s  modelName: %s  events: %d\n\n", podID, modelName, len(batch.Events))

	for i, event := range batch.Events {
		out, err := json.MarshalIndent(event, "", "    ")
		if err != nil {
			panic(err)
		}
		fmt.Printf("event[%d] (%T):\n%s\n\n", i, event, string(out))
	}
}
