// Copyright 2026 The llm-d Authors.
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

package kvevents

import (
	"testing"
	"time"
)

func residencyFor(t *testing.T, v *InMemorySessionView, tag, pod string) SessionResidency {
	t.Helper()
	for _, r := range v.Residency(tag) {
		if r.Pod == pod {
			return r
		}
	}
	return SessionResidency{}
}

func TestSessionViewFoldsContinuations(t *testing.T) {
	v := NewInMemorySessionView(0, 0)

	// Two continuation segments arrive in prefill order on pod-a.
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2, 3}, 48)
	v.AddBlocks("pod-a", "gpu", "s1", "c2", []uint64{4, 5}, 32)

	r := residencyFor(t, v, "s1", "pod-a")
	if r.UpTo != "c2" || r.Tokens != 80 {
		t.Fatalf("intact chain must cover both segments: %+v", r)
	}
}

func TestSessionViewSegmentExtension(t *testing.T) {
	// One continuation's blocks arriving across two events extend the same
	// segment rather than opening a new one.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2}, 32)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{3}, 16)

	r := residencyFor(t, v, "s1", "pod-a")
	if r.UpTo != "c1" || r.Tokens != 48 {
		t.Fatalf("same-continuation events must extend the segment: %+v", r)
	}
}

func TestSessionViewEvictionBreachesAndTruncates(t *testing.T) {
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2, 3}, 48)
	v.AddBlocks("pod-a", "gpu", "s1", "c2", []uint64{4, 5}, 32)
	v.AddBlocks("pod-a", "gpu", "s1", "c3", []uint64{6}, 16)

	// A block of c2 evicts: residency truncates to c1 — c3 survives
	// physically but is unreachable through an intact prefix.
	v.RemoveBlocks("pod-a", []uint64{5})

	r := residencyFor(t, v, "s1", "pod-a")
	if r.UpTo != "c1" || r.Tokens != 48 {
		t.Fatalf("eviction must truncate to the deepest intact continuation: %+v", r)
	}
}

func TestSessionViewEvictionOnOtherPodHarmless(t *testing.T) {
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2}, 32)
	v.AddBlocks("pod-b", "gpu", "s1", "c1", []uint64{1, 2}, 32)

	v.RemoveBlocks("pod-b", []uint64{1})

	if r := residencyFor(t, v, "s1", "pod-a"); r.UpTo != "c1" {
		t.Fatalf("pod-b eviction must not affect pod-a: %+v", r)
	}
	if r := residencyFor(t, v, "s1", "pod-b"); r.Tokens != 0 {
		t.Fatalf("pod-b residency must be gone: %+v", r)
	}
}

func TestSessionViewClearPod(t *testing.T) {
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1}, 16)
	v.AddBlocks("pod-b", "gpu", "s1", "c1", []uint64{2}, 16)

	v.ClearPod("pod-a")

	if r := residencyFor(t, v, "s1", "pod-a"); r.Tokens != 0 {
		t.Fatalf("cleared pod must report nothing: %+v", r)
	}
	if r := residencyFor(t, v, "s1", "pod-b"); r.UpTo != "c1" {
		t.Fatalf("other pod unaffected: %+v", r)
	}
}

func TestSessionViewSharedBlockBreachesAllSessions(t *testing.T) {
	// Content-addressed blocks are shared across sessions (forks, templates):
	// one eviction lowers every session referencing the block on that pod.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "parent", "c1", []uint64{1, 2}, 32)
	v.AddBlocks("pod-a", "gpu", "child", "c1", []uint64{1, 2}, 32)

	v.RemoveBlocks("pod-a", []uint64{2})

	if r := residencyFor(t, v, "parent", "pod-a"); r.Tokens != 0 {
		t.Fatalf("parent must be breached: %+v", r)
	}
	if r := residencyFor(t, v, "child", "pod-a"); r.Tokens != 0 {
		t.Fatalf("child must be breached: %+v", r)
	}
}

func TestSessionViewTTLExpiry(t *testing.T) {
	v := NewInMemorySessionView(time.Minute, 0)
	base := time.Unix(1_000_000, 0)
	v.now = func() time.Time { return base }
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1}, 16)

	v.now = func() time.Time { return base.Add(2 * time.Minute) }
	v.RemoveExpired()

	if r := v.Residency("s1"); len(r) != 0 {
		t.Fatalf("expired session must be dropped: %+v", r)
	}
}
