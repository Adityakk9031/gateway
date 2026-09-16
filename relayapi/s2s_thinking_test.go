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
func TestThinkingLevelsMatchProtocol(t *testing.T) {
	t.Parallel()
	for _, level := range []string{"", "low", "medium", "high", "minimal", "none", "HIGH", "xhigh"} {
		configure := relayapi.S2SSessionConfigure{
			Type:    relayapi.S2SControlSessionConfigure,
			Routing: relayapi.Routing{Mode: relayapi.RoutingModeExplicit, Provider: "google", Model: "gemini-3.8-live-extended-thinking"},
			Audio: relayapi.S2SAudioConfig{
				Input:  relayapi.AudioConfig{Encoding: "pcm_s16le", SampleRateHz: 16_000, Channels: 1},
				Output: relayapi.AudioConfig{Encoding: "pcm_s16le", SampleRateHz: 24_000, Channels: 1},
			},
			ThinkingLevel: level,
		}
		frameOK := configure.Validate() == nil
		planOK := protocol.ValidThinkingLevel(level)
		if frameOK != planOK {
			t.Fatalf("level %q: configure frame accepts=%v, plan accepts=%v", level, frameOK, planOK)
		}
	}
}
