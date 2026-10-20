package acp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

type promptScript struct {
	rateLimit bool // emit kiro rate_limit notification before the error
	update    bool // emit a session/update before the error (mid-turn failure)
	fail      bool // answer with -32603
}

// serveScripted answers each session/prompt according to scripts in order
// and records the prompt text of every request.
func serveScripted(rReq io.Reader, wResp io.Writer, scripts []promptScript) (*[]string, *sync.WaitGroup) {
	var mu sync.Mutex
	prompts := []string{}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sc := bufio.NewScanner(rReq)
		i := 0
		for sc.Scan() {
			var req struct {
				ID     json.RawMessage `json:"id"`
				Method string          `json:"method"`
				Params struct {
					Prompt []struct {
						Text string `json:"text"`
					} `json:"prompt"`
				} `json:"params"`
			}
			if json.Unmarshal(sc.Bytes(), &req) != nil || req.Method != "session/prompt" {
				continue
			}
			mu.Lock()
			prompts = append(prompts, req.Params.Prompt[0].Text)
			mu.Unlock()
			sc := scripts[min(i, len(scripts)-1)]
			i++
			if sc.update {
				_, _ = fmt.Fprint(wResp, `{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test-session-id","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"partial"}}}}`+"\n")
			}
			if sc.rateLimit {
				_, _ = fmt.Fprint(wResp, `{"jsonrpc":"2.0","method":"_kiro.dev/error/rate_limit","params":{}}`+"\n")
			}
			if sc.fail {
				_, _ = fmt.Fprintf(wResp, `{"jsonrpc":"2.0","id":%s,"error":{"code":-32603,"message":"Internal error"}}`+"\n", req.ID)
			} else {
				_, _ = fmt.Fprintf(wResp, `{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}`+"\n", req.ID)
			}
		}
	}()
	return &prompts, &wg
}

func fastBackoff(t *testing.T, n int) {
	t.Helper()
	orig := rateLimitBackoffs
	rateLimitBackoffs = make([]time.Duration, n)
	for i := range rateLimitBackoffs {
		rateLimitBackoffs[i] = time.Millisecond
	}
	t.Cleanup(func() { rateLimitBackoffs = orig })
}

func drainEvents(s *acpSession) []core.Event {
	var evs []core.Event
	for {
		select {
		case ev := <-s.events:
			evs = append(evs, ev)
		default:
			return evs
		}
	}
}

func TestSend_retriesAfterRateLimit(t *testing.T) {
	fastBackoff(t, 3)
	s, wResp, rReq := newTestSession(t, &fakeCallbacks{})
	prompts, _ := serveScripted(rReq, wResp, []promptScript{
		{rateLimit: true, fail: true},
		{rateLimit: true, fail: true},
		{},
	})

	if err := s.Send("hello", "m1", nil, nil); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(*prompts) != 3 {
		t.Fatalf("prompt calls = %d, want 3", len(*prompts))
	}
	for i, p := range *prompts {
		if p != "hello" {
			t.Fatalf("retry %d resent %q, want original prompt (no output had been produced)", i, p)
		}
	}
	for _, ev := range drainEvents(s) {
		if ev.Type == core.EventError {
			t.Fatalf("error event leaked to user during retried rate limit: %v", ev.Error)
		}
	}
}

func TestSend_rateLimitRetriesExhausted(t *testing.T) {
	fastBackoff(t, 2)
	s, wResp, rReq := newTestSession(t, &fakeCallbacks{})
	prompts, _ := serveScripted(rReq, wResp, []promptScript{{rateLimit: true, fail: true}})

	err := s.Send("hello", "m1", nil, nil)
	if err == nil {
		t.Fatal("Send should fail after retries are exhausted")
	}
	if len(*prompts) != 3 {
		t.Fatalf("prompt calls = %d, want 3 (1 + 2 retries)", len(*prompts))
	}
	errEvents := 0
	for _, ev := range drainEvents(s) {
		if ev.Type == core.EventError {
			errEvents++
		}
	}
	if errEvents != 1 {
		t.Fatalf("error events = %d, want exactly 1", errEvents)
	}
}

func TestSend_internalErrorWithoutRateLimitNotRetried(t *testing.T) {
	fastBackoff(t, 3)
	s, wResp, rReq := newTestSession(t, &fakeCallbacks{})
	prompts, _ := serveScripted(rReq, wResp, []promptScript{{fail: true}})

	if err := s.Send("hello", "m1", nil, nil); err == nil {
		t.Fatal("Send should surface a plain -32603")
	}
	if len(*prompts) != 1 {
		t.Fatalf("prompt calls = %d, want 1 (no rate_limit signal, no retry)", len(*prompts))
	}
}

// serveTrackingConcurrency answers every session/prompt after `hold`, while
// recording the highest number of requests it was handling simultaneously.
func serveTrackingConcurrency(rReq io.Reader, wResp io.Writer, hold time.Duration, inFlight, maxSeen *atomic.Int64) *sync.WaitGroup {
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sc := bufio.NewScanner(rReq)
		for sc.Scan() {
			var req struct {
				ID     json.RawMessage `json:"id"`
				Method string          `json:"method"`
			}
			if json.Unmarshal(sc.Bytes(), &req) != nil || req.Method != "session/prompt" {
				continue
			}
			go func(id json.RawMessage) {
				n := inFlight.Add(1)
				mu.Lock()
				if n > maxSeen.Load() {
					maxSeen.Store(n)
				}
				mu.Unlock()
				time.Sleep(hold)
				inFlight.Add(-1)
				_, _ = fmt.Fprintf(wResp, `{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}`+"\n", id)
			}(req.ID)
		}
	}()
	return &wg
}

func TestSend_maxConcurrentPromptsSerializesAcrossSessions(t *testing.T) {
	sem := make(chan struct{}, 1)
	var inFlight, maxSeen atomic.Int64

	s1, wResp1, rReq1 := newTestSession(t, &fakeCallbacks{})
	s1.promptSem = sem
	s2, wResp2, rReq2 := newTestSession(t, &fakeCallbacks{})
	s2.promptSem = sem
	s2.acpSessID = "test-session-id-2"

	serveTrackingConcurrency(rReq1, wResp1, 80*time.Millisecond, &inFlight, &maxSeen)
	serveTrackingConcurrency(rReq2, wResp2, 80*time.Millisecond, &inFlight, &maxSeen)

	var wg sync.WaitGroup
	wg.Add(2)
	var err1, err2 error
	go func() { defer wg.Done(); err1 = s1.Send("first", "m1", nil, nil) }()
	go func() { defer wg.Done(); err2 = s2.Send("second", "m2", nil, nil) }()
	wg.Wait()

	if err1 != nil {
		t.Fatalf("Send 1: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("Send 2: %v", err2)
	}
	if got := maxSeen.Load(); got != 1 {
		t.Fatalf("max concurrent server-side prompts = %d, want 1 (promptSem of size 1 should serialize)", got)
	}
}

// TestSend_hungCallForceReleasesSlotForSiblingSession reproduces the
// production incident: one session's session/prompt never gets *any*
// response (backend hang, unanswered permission request, dead subprocess
// that doesn't exit) and so never returns. Without a forced release, that
// permanently starves every sibling session sharing the same promptSem —
// which is exactly what happened when max_concurrent_prompts=1 was deployed.
func TestSend_hungCallForceReleasesSlotForSiblingSession(t *testing.T) {
	orig := promptSlotMaxHold
	promptSlotMaxHold = 30 * time.Millisecond
	t.Cleanup(func() { promptSlotMaxHold = orig })

	sem := make(chan struct{}, 1)
	var inFlight, maxSeen atomic.Int64

	hung, _, rReqHung := newTestSession(t, &fakeCallbacks{})
	hung.promptSem = sem
	ok, wResp2, rReq2 := newTestSession(t, &fakeCallbacks{})
	ok.promptSem = sem
	ok.acpSessID = "test-session-id-2"

	// hung's "server" never writes a response for any request — simulates a
	// backend that hangs forever, or a permission request nobody answers.
	go func() {
		_, _ = io.Copy(io.Discard, rReqHung)
	}()
	serveTrackingConcurrency(rReq2, wResp2, 10*time.Millisecond, &inFlight, &maxSeen)

	hungDone := make(chan error, 1)
	go func() { hungDone <- hung.Send("stuck forever", "m1", nil, nil) }()

	// Give the hung call time to acquire the slot and exceed promptSlotMaxHold.
	time.Sleep(150 * time.Millisecond)

	// The second, healthy session must still be able to get the slot and
	// complete — proving the force-release freed it rather than leaving it
	// permanently held.
	done2 := make(chan error, 1)
	go func() { done2 <- ok.Send("i should still work", "m2", nil, nil) }()

	select {
	case err := <-done2:
		if err != nil {
			t.Fatalf("sibling session's Send failed: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("sibling session never got the semaphore slot — hung call starved it forever")
	}

	// Clean up: the hung call is still blocked on its own RPC response
	// (that's realistic — it never completes on its own); cancel its
	// context so the test doesn't leak a goroutine.
	hung.cancel()
	<-hungDone
}

func TestSend_nilPromptSemAllowsConcurrentSends(t *testing.T) {
	var inFlight, maxSeen atomic.Int64

	s1, wResp1, rReq1 := newTestSession(t, &fakeCallbacks{})
	s2, wResp2, rReq2 := newTestSession(t, &fakeCallbacks{})
	s2.acpSessID = "test-session-id-2"
	// promptSem left nil on both — unlimited, matching every non-Kiro ACP
	// backend's pre-existing behaviour.

	serveTrackingConcurrency(rReq1, wResp1, 80*time.Millisecond, &inFlight, &maxSeen)
	serveTrackingConcurrency(rReq2, wResp2, 80*time.Millisecond, &inFlight, &maxSeen)

	var wg sync.WaitGroup
	wg.Add(2)
	var err1, err2 error
	go func() { defer wg.Done(); err1 = s1.Send("first", "m1", nil, nil) }()
	go func() { defer wg.Done(); err2 = s2.Send("second", "m2", nil, nil) }()
	wg.Wait()

	if err1 != nil {
		t.Fatalf("Send 1: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("Send 2: %v", err2)
	}
	if got := maxSeen.Load(); got != 2 {
		t.Fatalf("max concurrent server-side prompts = %d, want 2 (nil promptSem must not serialize)", got)
	}
}

func TestSend_midTurnRateLimitSendsContinuePrompt(t *testing.T) {
	fastBackoff(t, 3)
	s, wResp, rReq := newTestSession(t, &fakeCallbacks{})
	prompts, _ := serveScripted(rReq, wResp, []promptScript{
		{update: true, rateLimit: true, fail: true},
		{},
	})

	if err := s.Send("do the big task", "m1", nil, nil); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(*prompts) != 2 {
		t.Fatalf("prompt calls = %d, want 2", len(*prompts))
	}
	if (*prompts)[1] == "do the big task" {
		t.Fatal("mid-turn retry must not replay the original prompt (would redo finished work)")
	}
	if (*prompts)[1] != rateLimitContinuePrompt {
		t.Fatalf("retry prompt = %q, want continue prompt", (*prompts)[1])
	}
}
