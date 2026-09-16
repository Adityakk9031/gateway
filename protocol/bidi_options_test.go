package protocol

import (
	"encoding/json"
	"strings"
	"testing"
)

// The setup carrier is reachable through two entry points — S2SOptions on a
// session plan request and VoiceSessionConfigure on the voice route — and both
// must hold it to the same bound, or the restriction would depend on which
// door the document came through.
func TestS2SOptionsValidateTheBidiSetupToo(t *testing.T) {
	t.Parallel()
	media := MediaFormat{Encoding: "pcm_s16le", SampleRateHz: 24_000, Channels: 1}
	for name, setup := range map[string]json.RawMessage{
		"not an object": json.RawMessage(`[]`),
		"oversized":     json.RawMessage(`{"pad":"` + strings.Repeat("x", MaxBidiSetupBytes) + `"}`),
	} {
		options := S2SOptions{OutputMedia: &media, Bidi: &BidiOptions{Setup: setup}}
		if err := options.validate(); err == nil {
			t.Fatalf("%s: S2SOptions accepted a setup BidiOptions.Validate refuses", name)
		}
		configure := VoiceSessionConfigure{Protocol: SpeechProtocolGoogleLiveV1, Media: media, OutputMedia: media, Bidi: &BidiOptions{Setup: setup}}
		if err := configure.Validate(); err == nil {
			t.Fatalf("%s: VoiceSessionConfigure accepted a setup BidiOptions.Validate refuses", name)
		}
	}
	valid := S2SOptions{OutputMedia: &media, Bidi: &BidiOptions{Setup: json.RawMessage(`{"model":"gemini-3.8-live"}`)}}
	if err := valid.validate(); err != nil {
		t.Fatalf("a bounded object setup must pass: %v", err)
	}
}
