package relayapi_test

import (
	"testing"

	"github.com/SpekoAI/gateway/protocol"
	"github.com/SpekoAI/gateway/relayapi"
)

// The public configure frame and the signed plan must accept the SAME thinking
// levels. relayapi cannot import protocol (doc.go: standard library only), so
// the set is written twice — and drift between the two would let the edge
// accept a level admission then rejects, or the reverse.
//
// The sets are compared directly rather than sampled: a level added to only
// one side is exactly the drift this test exists to catch, and a fixed list of
// values would never see it.
func TestThinkingLevelsMatchProtocol(t *testing.T) {
	t.Parallel()
	frame := relayapi.ThinkingLevels()
	plan := protocol.ThinkingLevels()
	if len(frame) != len(plan) {
		t.Fatalf("thinking levels differ: relayapi=%v protocol=%v", frame, plan)
	}
	for i := range frame {
		if frame[i] != plan[i] {
			t.Fatalf("thinking levels differ at %d: relayapi=%v protocol=%v", i, frame, plan)
		}
	}
}

// And the validators must agree with the sets they publish, including on the
// values a caller is most likely to try (`minimal` is rejected by the service
// even on the extended-thinking model, so it must be rejected here too).
func TestThinkingLevelValidatorsAgree(t *testing.T) {
	t.Parallel()
	audio := relayapi.S2SAudioConfig{
		Input:  relayapi.AudioConfig{Encoding: "pcm_s16le", SampleRateHz: 16_000, Channels: 1},
		Output: relayapi.AudioConfig{Encoding: "pcm_s16le", SampleRateHz: 24_000, Channels: 1},
	}
	candidates := append(relayapi.ThinkingLevels(), "", "minimal", "none", "xhigh", "HIGH", "max")
	for _, level := range candidates {
		configure := relayapi.S2SSessionConfigure{
			Type: relayapi.S2SControlSessionConfigure,
			Routing: relayapi.Routing{
				Mode:     relayapi.RoutingModeExplicit,
				Provider: "google",
				Model:    "gemini-3.8-live-extended-thinking",
			},
			Audio:         audio,
			ThinkingLevel: level,
		}
		frameOK := configure.Validate() == nil
		planOK := protocol.ValidThinkingLevel(level)
		if frameOK != planOK {
			t.Fatalf("level %q: configure frame accepts=%v, plan accepts=%v", level, frameOK, planOK)
		}
	}
}
