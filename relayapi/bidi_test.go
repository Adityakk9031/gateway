package relayapi_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/SpekoAI/gateway/relayapi"
)

func validBidiSetup(t *testing.T) relayapi.BidiSetup {
	t.Helper()
	setup, err := relayapi.DecodeBidiSetup([]byte(`{"setup":{"model":"gemini-3.8-live"}}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return setup
}

func TestBidiSetupAcceptsBothModelSpellings(t *testing.T) {
	t.Parallel()
	// The vendor's own snippets say "models/<id>"; the catalog and the signed
	// plan say the bare id. Both must land on the same admitted model or a
	// copy-pasted example fails at the Router for no transport reason.
	for _, raw := range []string{
		`{"setup":{"model":"gemini-3.8-live"}}`,
		`{"setup":{"model":"models/gemini-3.8-live"}}`,
	} {
		setup, err := relayapi.DecodeBidiSetup([]byte(raw))
		if err != nil {
			t.Fatalf("%s: decode: %v", raw, err)
		}
		if err := setup.Validate(); err != nil {
			t.Fatalf("%s: validate: %v", raw, err)
		}
		if got := setup.Setup.ModelID(); got != "gemini-3.8-live" {
			t.Fatalf("%s: ModelID = %q", raw, got)
		}
	}
}

func TestBidiSetupRejectsUnknownFields(t *testing.T) {
	t.Parallel()
	// Strict decoding is the promise that the Router forwarded everything it
	// accepted — a silently dropped field would surface as the model ignoring
	// a setting the caller believes it sent.
	if _, err := relayapi.DecodeBidiSetup([]byte(`{"setup":{"model":"gemini-3.8-live","nope":1}}`)); err == nil {
		t.Fatal("expected an unknown setup field to be refused")
	}
	if _, err := relayapi.DecodeBidiSetup([]byte(`{"setup":{"model":"x"}} {"setup":{}}`)); err == nil {
		t.Fatal("expected trailing content to be refused")
	}
}

func TestBidiSetupRejectsAutoAndMalformedModels(t *testing.T) {
	t.Parallel()
	for _, model := range []string{"", "auto", "models/", "gemini live", strings.Repeat("m", 129)} {
		setup := relayapi.BidiSetup{}
		setup.Setup.Model = model
		if err := setup.Validate(); err == nil {
			t.Fatalf("model %q should be refused", model)
		}
	}
}

func TestBidiSetupBoundsOversizedDocuments(t *testing.T) {
	t.Parallel()
	setup := validBidiSetup(t)
	setup.Setup.SystemInstruction = json.RawMessage(`"` + strings.Repeat("a", relayapi.MaxBidiInstructionsBytes) + `"`)
	assertInvalid(t, setup.Validate(), "systemInstruction")

	setup = validBidiSetup(t)
	setup.Setup.RealtimeInputConfig = json.RawMessage(`{"pad":"` + strings.Repeat("a", relayapi.MaxBidiSettingBytes) + `"}`)
	assertInvalid(t, setup.Validate(), "realtimeInputConfig")

	setup = validBidiSetup(t)
	tools := make([]json.RawMessage, relayapi.MaxBidiTools+1)
	for i := range tools {
		tools[i] = json.RawMessage(`{}`)
	}
	setup.Setup.Tools = tools
	assertInvalid(t, setup.Validate(), "tools")
}

func TestBidiMessageKeyClassifiesByTopLevelKey(t *testing.T) {
	t.Parallel()
	// The key IS the message type here; there is no "type" field to read.
	for raw, want := range map[string]string{
		`{"setup":{}}`:         relayapi.BidiSetupKey,
		`{"realtimeInput":{}}`: relayapi.BidiRealtimeInputKey,
		`{"clientContent":{}}`: "clientContent",
		`{"toolResponse":{}}`:  "toolResponse",
	} {
		got, ok := relayapi.BidiMessageKey([]byte(raw))
		if !ok || got != want {
			t.Fatalf("%s: got (%q,%v), want %q", raw, got, ok, want)
		}
	}
	// Zero keys and several keys are both unclassifiable, and guessing which
	// one the caller meant would forward the wrong message.
	for _, raw := range []string{`{}`, `{"a":1,"b":2}`, `[]`, `not json`, `{"":1}`} {
		if _, ok := relayapi.BidiMessageKey([]byte(raw)); ok {
			t.Fatalf("%s should not classify", raw)
		}
	}
}

func TestDecodeBidiAudioAcceptsBothVendorShapes(t *testing.T) {
	t.Parallel()
	// `mediaChunks` is the older spelling and the vendor's SDKs still emit it.
	chunks, end, ok := relayapi.DecodeBidiAudio([]byte(`{"realtimeInput":{"audio":{"data":"AAA="}}}`))
	if !ok || end || len(chunks) != 1 || chunks[0] != "AAA=" {
		t.Fatalf("audio: got (%v,%v,%v)", chunks, end, ok)
	}
	chunks, _, ok = relayapi.DecodeBidiAudio([]byte(`{"realtimeInput":{"mediaChunks":[{"data":"AAA="},{"data":"BBB="}]}}`))
	if !ok || len(chunks) != 2 {
		t.Fatalf("mediaChunks: got (%v,%v)", chunks, ok)
	}
	if _, end, ok = relayapi.DecodeBidiAudio([]byte(`{"realtimeInput":{"audioStreamEnd":true}}`)); !ok || !end {
		t.Fatalf("audioStreamEnd: got (%v,%v)", end, ok)
	}
	if _, _, ok = relayapi.DecodeBidiAudio([]byte(`{"realtimeInput":{}}`)); ok {
		t.Fatal("an empty realtimeInput carries nothing and must not read as audio")
	}
}

func TestBidiRouteIsDistinctFromTheOpenAIRoutes(t *testing.T) {
	t.Parallel()
	// Three protocols, three paths: a collision would route a session to a
	// socket that speaks a different framing.
	paths := map[string]bool{
		relayapi.RealtimeRoutePath: true,
		relayapi.LiveRoutePath:     true,
		relayapi.BidiRoutePath:     true,
	}
	if len(paths) != 3 {
		t.Fatalf("voice route paths collide: %v", paths)
	}
}
