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
	"sync"
	"time"
)

// SessionView is a session-granular projection over the same KV events that
// feed the block index: blocks arrive labeled with an opaque (sessionTag,
// continuationID) pair, and evictions fold through the mapping so that a
// session's residency on a pod is always "the deepest continuation whose
// blocks all survive". The view interprets neither tag — continuation
// ordering is arrival order, which per-pod worker sharding guarantees to be
// prefill order.
type SessionView interface {
	// AddBlocks records blocks stored for (sessionTag, continuationID) on a
	// pod. tokenCount is the number of tokens the stored blocks cover, in
	// engine units.
	AddBlocks(pod, tier, sessionTag, continuationID string, blockHashes []uint64, tokenCount int)
	// RemoveBlocks folds evictions: any continuation segment containing a
	// removed block is breached, truncating its session's high-water.
	RemoveBlocks(pod string, blockHashes []uint64)
	// ClearPod drops all state for a pod (engine restart, cache clear).
	ClearPod(pod string)
	// Residency reports, per pod, the deepest intact continuation and the
	// engine-unit token count covered by the intact prefix.
	Residency(sessionTag string) []SessionResidency
}

// SessionResidency is one pod's residency for a session.
type SessionResidency struct {
	// Pod is the pod identifier the residency describes.
	Pod string
	// Tier is the device tier of the deepest intact segment.
	Tier string
	// UpTo is the deepest continuation id whose prefix is fully resident.
	UpTo string
	// Tokens is the engine-unit token count covered by the intact prefix.
	Tokens int
}

// segment is one continuation's block extent on one pod.
type segment struct {
	sessionTag     string
	continuationID string
	pod            string
	tier           string
	total          int
	alive          int
	tokens         int
	hashes         []uint64
}

func (s *segment) intact() bool { return s.alive == s.total }

// podChain is the ordered list of segments for one (session, pod).
type podChain struct {
	segments []*segment
}

type sessionState struct {
	pods     map[string]*podChain
	lastSeen time.Time
}

// InMemorySessionView is the default SessionView implementation. It is safe
// for concurrent use: pool workers write under a mutex while readers
// (Residency, PodMass) take a read lock.
type InMemorySessionView struct {
	mu       sync.RWMutex
	sessions map[string]*sessionState
	// blockRef maps a block hash to every segment referencing it on any pod;
	// one eviction breaches all of them.
	blockRef map[uint64][]*segment

	ttl         time.Duration
	maxSessions int
	now         func() time.Time
}

const (
	defaultSessionViewTTL = 30 * time.Minute
	defaultSessionViewCap = 100_000
	sessionViewEvictBatch = 128
)

// NewInMemorySessionView returns a SessionView with the given idle TTL and
// session cap; zero values select defaults.
func NewInMemorySessionView(ttl time.Duration, maxSessions int) *InMemorySessionView {
	if ttl <= 0 {
		ttl = defaultSessionViewTTL
	}
	if maxSessions <= 0 {
		maxSessions = defaultSessionViewCap
	}
	return &InMemorySessionView{
		sessions:    map[string]*sessionState{},
		blockRef:    map[uint64][]*segment{},
		ttl:         ttl,
		maxSessions: maxSessions,
		now:         time.Now,
	}
}

// AddBlocks implements SessionView.
func (v *InMemorySessionView) AddBlocks(pod, tier, sessionTag, continuationID string, blockHashes []uint64, tokenCount int) {
	if sessionTag == "" || pod == "" || len(blockHashes) == 0 {
		return
	}
	now := v.now()

	v.mu.Lock()
	defer v.mu.Unlock()

	state, ok := v.sessions[sessionTag]
	if !ok {
		if len(v.sessions) >= v.maxSessions {
			v.evictLocked(now)
		}
		state = &sessionState{pods: map[string]*podChain{}}
		v.sessions[sessionTag] = state
	}
	state.lastSeen = now

	chain, ok := state.pods[pod]
	if !ok {
		chain = &podChain{}
		state.pods[pod] = chain
	}

	// Re-admission heals: blocks are content-addressed, so a hash stored
	// again on this pod after an eviction is the SAME block coming back.
	// Every breached segment referencing it on this pod regains a life —
	// without healing, one evict-and-reprefill cycle would truncate the
	// session's residency forever and turn engine truth into false-cold.
	// A hash that heals THIS session's own segment is not re-recorded (the
	// chain position already accounts for it); a hash that only heals other
	// sessions' segments still gets recorded under this session's chain.
	fresh := blockHashes[:0:0]
	for _, h := range blockHashes {
		healedOwn := false
		for _, ref := range v.blockRef[h] {
			if ref.pod == pod && ref.alive < ref.total {
				ref.alive++
				if ref.sessionTag == sessionTag {
					healedOwn = true
				}
			}
		}
		if !healedOwn {
			fresh = append(fresh, h)
		}
	}
	if len(fresh) == 0 {
		return
	}

	// Same continuation as the tail segment: extend it (one continuation's
	// blocks may arrive across several events). Otherwise a new segment.
	var seg *segment
	if n := len(chain.segments); n > 0 && chain.segments[n-1].continuationID == continuationID {
		seg = chain.segments[n-1]
	} else {
		seg = &segment{sessionTag: sessionTag, continuationID: continuationID, pod: pod, tier: tier}
		chain.segments = append(chain.segments, seg)
	}
	seg.total += len(fresh)
	seg.alive += len(fresh)
	seg.tokens += tokenCount
	if tier != "" {
		seg.tier = tier
	}
	seg.hashes = append(seg.hashes, fresh...)
	for _, h := range fresh {
		v.blockRef[h] = append(v.blockRef[h], seg)
	}
}

// RemoveBlocks implements SessionView.
func (v *InMemorySessionView) RemoveBlocks(pod string, blockHashes []uint64) {
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, h := range blockHashes {
		for _, seg := range v.blockRef[h] {
			// Breach: the segment loses a block. alive is a count, not a
			// per-hash ledger — per-pod event ordering guarantees stores and
			// removals of one hash alternate, so one removal maps to one
			// stored block. The reference is KEPT: a breached segment must
			// stay reachable so re-admission of the same content hash can
			// heal it (segments are dropped only with their session or pod).
			if seg.pod == pod && seg.alive > 0 {
				seg.alive--
			}
		}
	}
}

// ClearPod implements SessionView.
func (v *InMemorySessionView) ClearPod(pod string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, state := range v.sessions {
		if chain, ok := state.pods[pod]; ok {
			for _, seg := range chain.segments {
				v.unrefLocked(seg)
			}
			delete(state.pods, pod)
		}
	}
}

// Residency implements SessionView.
func (v *InMemorySessionView) Residency(sessionTag string) []SessionResidency {
	v.mu.RLock()
	defer v.mu.RUnlock()
	state, ok := v.sessions[sessionTag]
	if !ok {
		return nil
	}
	out := make([]SessionResidency, 0, len(state.pods))
	for pod, chain := range state.pods {
		res := SessionResidency{Pod: pod}
		for _, seg := range chain.segments {
			if !seg.intact() {
				break
			}
			res.UpTo = seg.continuationID
			res.Tokens += seg.tokens
			res.Tier = seg.tier
		}
		if res.Tokens > 0 {
			out = append(out, res)
		}
	}
	return out
}

// RemoveExpired drops sessions idle past the TTL. The view runs no background
// sweeper of its own; callers invoke it periodically. Independently, an
// insertion at capacity reaps expired sessions before evicting live ones.
func (v *InMemorySessionView) RemoveExpired() {
	now := v.now()
	v.mu.Lock()
	defer v.mu.Unlock()
	cutoff := now.Add(-v.ttl)
	for tag, state := range v.sessions {
		if state.lastSeen.Before(cutoff) {
			v.dropLocked(tag, state)
		}
	}
}

// evictLocked drops expired sessions and, if still at capacity, arbitrary
// ones (bounded). Callers must hold v.mu.
func (v *InMemorySessionView) evictLocked(now time.Time) {
	cutoff := now.Add(-v.ttl)
	for tag, state := range v.sessions {
		if state.lastSeen.Before(cutoff) {
			v.dropLocked(tag, state)
		}
	}
	if len(v.sessions) < v.maxSessions {
		return
	}
	dropped := 0
	for tag, state := range v.sessions {
		v.dropLocked(tag, state)
		dropped++
		if dropped >= sessionViewEvictBatch {
			break
		}
	}
}

// dropLocked removes a session and its block references. Callers must hold v.mu.
func (v *InMemorySessionView) dropLocked(tag string, state *sessionState) {
	for _, chain := range state.pods {
		for _, seg := range chain.segments {
			v.unrefLocked(seg)
		}
	}
	delete(v.sessions, tag)
}

// unrefLocked removes a segment from the block reference map. Callers must
// hold v.mu.
func (v *InMemorySessionView) unrefLocked(seg *segment) {
	for _, h := range seg.hashes {
		refs := v.blockRef[h]
		kept := refs[:0]
		for _, s := range refs {
			if s != seg {
				kept = append(kept, s)
			}
		}
		if len(kept) == 0 {
			delete(v.blockRef, h)
		} else {
			v.blockRef[h] = kept
		}
	}
}
