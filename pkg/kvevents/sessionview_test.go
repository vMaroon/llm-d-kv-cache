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

func TestSessionViewReadmissionHeals(t *testing.T) {
	// Evict-and-reprefill: the same content hashes stored again must heal
	// the breached segments, or one churn cycle turns residency false-cold
	// forever.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2, 3}, 48)
	v.AddBlocks("pod-a", "gpu", "s1", "c2", []uint64{4, 5}, 32)

	v.RemoveBlocks("pod-a", []uint64{2, 4})
	if r := residencyFor(t, v, "s1", "pod-a"); r.Tokens != 0 {
		t.Fatalf("post-eviction residency must be truncated: %+v", r)
	}

	// The next turn re-prefills the evicted blocks (same content hashes).
	v.AddBlocks("pod-a", "gpu", "s1", "c3", []uint64{2, 4, 6}, 24)

	r := residencyFor(t, v, "s1", "pod-a")
	if r.UpTo != "c3" || r.Tokens != 48+32+24 {
		t.Fatalf("re-admission must heal the chain end to end: %+v", r)
	}
}

func TestSessionViewHealOtherSessionStillRecordsOwn(t *testing.T) {
	// A block healing another session's breached segment still counts for
	// the storing session's own chain (shared templates across sessions).
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "old", "c1", []uint64{1, 2}, 32)
	v.RemoveBlocks("pod-a", []uint64{1})

	v.AddBlocks("pod-a", "gpu", "new", "c1", []uint64{1, 2}, 32)

	if r := residencyFor(t, v, "new", "pod-a"); r.Tokens != 32 {
		t.Fatalf("storing session must keep its own claim: %+v", r)
	}
	if r := residencyFor(t, v, "old", "pod-a"); r.Tokens != 32 {
		t.Fatalf("breached session must be healed by re-admission: %+v", r)
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

func TestSessionViewPodMass(t *testing.T) {
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("pod-a", "gpu", "s1", "c1", []uint64{1, 2}, 32)
	v.AddBlocks("pod-a", "gpu", "s2", "c1", []uint64{3}, 16)
	v.AddBlocks("pod-b", "gpu", "s1", "c1", []uint64{4}, 16)

	m := v.PodMass()
	if m["pod-a"] != 48 || m["pod-b"] != 16 {
		t.Fatalf("mass = %v, want pod-a 48, pod-b 16", m)
	}

	// A breach removes the segment (and everything after it) from the mass.
	v.RemoveBlocks("pod-a", []uint64{1})
	m = v.PodMass()
	if m["pod-a"] != 16 {
		t.Fatalf("post-breach mass = %v, want pod-a 16 (s2 only)", m)
	}
}

func TestChainSurvivalCrossLineage(t *testing.T) {
	// A re-minted or forked lineage shares continuation ids for content
	// before its divergence; the chain lookup must credit the parent's
	// stored prefix, whoever stored it.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("podA", "gpu", "parent", "c1", []uint64{1, 2}, 3200)
	v.AddBlocks("podA", "gpu", "parent", "c2", []uint64{3}, 800)

	got := v.LongestSurvivingPrefix([]string{"c1", "c2", "c9"})
	s, ok := got["podA"]
	if !ok || !s.Known {
		t.Fatalf("chain sharing c1,c2 must be known on podA: %+v", got)
	}
	if s.Tokens != 4000 {
		t.Fatalf("survival through c2 = 4000 tokens, got %d", s.Tokens)
	}

	// A chain diverging after c1 only credits through c1.
	got = v.LongestSurvivingPrefix([]string{"c1", "x2", "x3"})
	if s := got["podA"]; s.Tokens != 3200 {
		t.Fatalf("survival through c1 = 3200 tokens, got %d", s.Tokens)
	}
}

func TestChainSurvivalBreachIsKnownCold(t *testing.T) {
	// Eviction breaches the stored chain: survival drops, but the pod stays
	// Known — confirmed-cold, not unknown. Healing restores it.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("podA", "gpu", "s", "c1", []uint64{1, 2}, 3200)
	v.RemoveBlocks("podA", []uint64{1})

	got := v.LongestSurvivingPrefix([]string{"c1", "c2"})
	s := got["podA"]
	if !s.Known || s.Tokens != 0 {
		t.Fatalf("breached chain must be Known with zero survival: %+v", s)
	}

	v.AddBlocks("podA", "gpu", "other", "c1", []uint64{1}, 1600)
	got = v.LongestSurvivingPrefix([]string{"c1"})
	if s := got["podA"]; s.Tokens != 3200 {
		t.Fatalf("healed chain must recover full survival, got %d", s.Tokens)
	}
}

func TestChainSurvivalSumsAcrossReMintedTags(t *testing.T) {
	// The re-mint case the fix exists for: turns 1-2 stored under the original
	// tag, turn 3 re-minted to a new tag owning only its own suffix segment.
	// Survival must sum the matched heads across tags to reconstruct the full
	// prefix, not return the deep re-mint's tiny suffix alone.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("podA", "gpu", "t1", "c1", []uint64{1}, 1000)
	v.AddBlocks("podA", "gpu", "t1", "c2", []uint64{2}, 1000)
	v.AddBlocks("podA", "gpu", "t2", "c3", []uint64{3}, 1000) // re-mint owns only c3

	got := v.LongestSurvivingPrefix([]string{"c1", "c2", "c3", "c9"})
	if s := got["podA"]; s.Tokens != 3000 {
		t.Fatalf("re-mint must credit the ancestor prefix across tags: got %d, want 3000", s.Tokens)
	}

	// Dropping the ancestor tag drops its contribution; the re-mint's own
	// suffix still resolves.
	v.sessions["t1"].lastSeen = v.now().Add(-time.Hour)
	v.RemoveExpired()
	got = v.LongestSurvivingPrefix([]string{"c1", "c2", "c3"})
	if s := got["podA"]; s.Tokens != 1000 {
		t.Fatalf("after ancestor drop only the re-mint suffix remains: got %d, want 1000", s.Tokens)
	}
}

func TestChainSurvivalBreaksAtMiddleBreach(t *testing.T) {
	// An evicted middle head truncates the contiguous prefix even if deeper
	// heads survive: the engine can only reuse up to the gap.
	v := NewInMemorySessionView(0, 0)
	v.AddBlocks("podA", "gpu", "t", "c1", []uint64{1}, 1000)
	v.AddBlocks("podA", "gpu", "t", "c2", []uint64{2}, 1000)
	v.AddBlocks("podA", "gpu", "t", "c3", []uint64{3}, 1000)
	v.RemoveBlocks("podA", []uint64{2}) // evict turn-2 block -> c2 breached

	got := v.LongestSurvivingPrefix([]string{"c1", "c2", "c3"})
	s := got["podA"]
	if !s.Known {
		t.Fatal("a breached prefix must stay Known")
	}
	if s.Tokens != 1000 {
		t.Fatalf("middle breach truncates to the surviving prefix: got %d, want 1000", s.Tokens)
	}
}
