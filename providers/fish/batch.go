package fish

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/SpekoAI/gateway/internal/batchhttp"
	"github.com/SpekoAI/gateway/internal/upstream"
	"github.com/SpekoAI/gateway/metering"
	runtimepkg "github.com/SpekoAI/gateway/runtime"
)

const (
	// BatchAdapterID identifies the Fish Audio file-transcription implementation.
	BatchAdapterID = "fish.stt.batch.v1"
	// BatchEndpoint is Fish's synchronous ASR endpoint. Both models share it;
	// the model travels in the `model` HTTP header, never in the body.
	BatchEndpoint = "https://api.fish.audio/v1/asr"
	// BatchModel is the multi-speaker model: inline <|speaker:N|> markers and
	// emotion/vocal-event cues ([laughter]) in the text.
	BatchModel = "transcribe-1-pro"
	// BatchModelStandard is the general model Fish serves when the header is
	// omitted.
	BatchModelStandard = "transcribe-1"
	// BatchMaxDurationSeconds is deliberately below what Fish accepts. The
	// service refuses more than 600 s (400 APEX_ASR_MAX_AUDIO_SECONDS), but
	// measured on 2026-09-28 it silently returns no segments from roughly
	// 290 s, and transcribe-1-pro truncates its text at 5040 characters. With
	// no segments the relay cannot detect a truncated transcript, so the cap
	// keeps every request inside the range where alignment comes back; longer
	// audio is chunked by the jobs path.
	BatchMaxDurationSeconds int64 = 240
	// BatchMaxAudioBytes is the measured body ceiling (a 44 MiB upload passed,
	// 52 MiB answered 413) less room for the multipart envelope.
	BatchMaxAudioBytes int64 = (44 << 20) - (16 << 10)

	batchExtensionID = "fish.audio/v1/asr"
	// proTextCapRunes is where transcribe-1-pro cut every long transcript
	// measured, mid-marker included, while still answering 200.
	proTextCapRunes = 5040
)

var batchModels = map[string]struct{}{BatchModel: {}, BatchModelStandard: {}}

var (
	speakerMarker = regexp.MustCompile(`<\|speaker:(\d+)\|>`)
	// cue matches the bracketed emotion and vocal-event annotations Pro keeps
	// in the text. They are transcript content, but Fish excludes them from
	// timestamp alignment, so alignment must too.
	cue = regexp.MustCompile(`\[[^\[\]]*\]`)
)

// BatchConfig controls local transport limits for the ASR adapter.
type BatchConfig struct {
	AdapterID             string
	HTTPClient            *http.Client
	MaxResponseBytes      int64
	AllowedEndpointHosts  []string
	AllowInsecureEndpoint bool
}

// BatchAdapter implements runtime.BatchTranscriber over POST /v1/asr: one
// multipart request, one JSON response.
type BatchAdapter struct {
	id               string
	httpClient       *http.Client
	maxResponseBytes int64
	endpointPolicy   upstream.HTTPPolicy
}

// NewBatch creates the Fish Audio ASR adapter.
func NewBatch(config BatchConfig) (*BatchAdapter, error) {
	if config.AdapterID == "" {
		config.AdapterID = BatchAdapterID
	}
	if config.MaxResponseBytes == 0 {
		config.MaxResponseBytes = batchhttp.DefaultMaxResponseBytes
	}
	if config.MaxResponseBytes < 1 {
		return nil, errors.New("fish batch maximum response bytes must be positive")
	}
	policy, err := upstream.NewHTTPPolicy(officialAPIHost, config.AllowedEndpointHosts, config.AllowInsecureEndpoint)
	if err != nil {
		return nil, err
	}
	return &BatchAdapter{id: config.AdapterID, httpClient: config.HTTPClient, maxResponseBytes: config.MaxResponseBytes, endpointPolicy: policy}, nil
}

func (a *BatchAdapter) ID() string { return a.id }

// Transcribe POSTs the WAV as the multipart `audio` part with timestamps on.
// Timestamps are always requested because the segments are the only evidence
// the relay has that the transcript covers the whole upload.
func (a *BatchAdapter) Transcribe(ctx context.Context, request runtimepkg.BatchTranscribeRequest) (*runtimepkg.BatchTranscription, error) {
	if request.Plan.Route.Provider != "fish" {
		return nil, fmt.Errorf("fish batch adapter cannot serve provider %q", request.Plan.Route.Provider)
	}
	// An unknown `model` header is not rejected: Fish silently runs
	// transcribe-1 and bills it. Only the two published ids leave here.
	model := strings.TrimSpace(request.Plan.Route.Model)
	if _, ok := batchModels[model]; !ok {
		return nil, fmt.Errorf("fish batch adapter requires %s or %s, got %q", BatchModelStandard, BatchModel, model)
	}
	if request.AudioBytes > BatchMaxAudioBytes {
		return nil, &runtimepkg.ProviderError{Code: batchhttp.CodeInputTooLarge, Message: "the upload exceeds Fish Audio's ASR body limit"}
	}
	credential, err := batchhttp.Credential(request.Plan)
	if err != nil {
		return nil, err
	}
	endpoint, err := a.endpointPolicy.Parse(request.Plan.Route.Endpoint)
	if err != nil {
		return nil, err
	}
	fields := []batchhttp.MultipartField{{Name: "ignore_timestamps", Value: "false"}}
	// The language is a hint: detection still runs and an unknown code is
	// ignored rather than refused.
	if language := baseLanguage(request.Options.Language); language != "" {
		fields = append(fields, batchhttp.MultipartField{Name: "language", Value: language})
	}
	body, contentType := batchhttp.Multipart(fields, "audio", "audio.wav", "audio/wav", request.Audio)
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), body)
	if err != nil {
		body.Close()
		return nil, err
	}
	httpRequest.Header.Set("Content-Type", contentType)
	httpRequest.Header.Set("Accept", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+credential)
	httpRequest.Header.Set("model", model)

	response, err := batchhttp.Do(a.httpClient, httpRequest, a.maxResponseBytes)
	if err != nil {
		return nil, err
	}
	if response.Status < 200 || response.Status >= 300 {
		return nil, batchhttp.StatusError(batchExtensionID, response.Status, response.Body)
	}
	var payload struct {
		Text         string  `json:"text"`
		Duration     float64 `json:"duration"`
		LanguageCode string  `json:"language_code"`
		Segments     []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"segments"`
	}
	if err := batchhttp.DecodeJSON(response.Body, &payload); err != nil {
		return nil, err
	}
	turns := parseTurns(payload.Text)
	text := joinTurns(turns)
	// Fish's segments are single words. They carry no speaker; the speaker is
	// recovered from the marker turns in text.
	words := make([]batchhttp.Word, 0, len(payload.Segments))
	for _, segment := range payload.Segments {
		if strings.TrimSpace(segment.Text) == "" {
			continue
		}
		words = append(words, batchhttp.Word{Text: strings.TrimSpace(segment.Text), StartMS: batchhttp.SecondsToMS(segment.Start), EndMS: batchhttp.SecondsToMS(segment.End)})
	}
	if text != "" && len(words) == 0 {
		// Text with no alignment cannot be checked against the audio it
		// claims to cover, and a truncated transcript looks exactly like this.
		return nil, batchhttp.Malformed(errors.New("fish transcript carries no timestamped segments"))
	}
	if model == BatchModel && utf8.RuneCountInString(payload.Text) >= proTextCapRunes {
		return nil, batchhttp.Malformed(fmt.Errorf("fish transcript reached the %d-character cap and is truncated", proTextCapRunes))
	}
	diarize := request.Options.STT.Diarize()
	segments, aligned := alignedSegments(turns, words, diarize)
	if !aligned {
		segments = batchhttp.GroupWords(words, 0)
	}
	requestID := response.Header.Get("x-fish-trace-id")
	observation := metering.Duration("request", model, "batch", response.Body, 1000, "duration")
	observation.ProviderRequestID = requestID
	result := &runtimepkg.BatchTranscription{
		Billing:           metering.Report(observation),
		Text:              text,
		Segments:          segments,
		Language:          payload.LanguageCode,
		DurationMS:        batchhttp.SecondsToMS(payload.Duration),
		ProviderRequestID: requestID,
		Extensions:        batchhttp.RawExtension(batchExtensionID, response.Body),
	}
	if request.Options.STT.WantsWordTimestamps() {
		for _, word := range words {
			result.Words = append(result.Words, runtimepkg.BatchWord{Text: word.Text, StartMS: word.StartMS, EndMS: word.EndMS, Speaker: word.Speaker})
		}
	}
	return result, nil
}

// turn is one speaker's stretch of the transcript. Speaker is empty for
// transcribe-1, which emits no markers.
type turn struct {
	speaker string
	text    string
}

func parseTurns(raw string) []turn {
	var turns []turn
	add := func(speaker, text string) {
		if text = strings.Join(strings.Fields(text), " "); text != "" {
			turns = append(turns, turn{speaker: speaker, text: text})
		}
	}
	speaker, last := "", 0
	for _, match := range speakerMarker.FindAllStringSubmatchIndex(raw, -1) {
		add(speaker, raw[last:match[0]])
		speaker, last = raw[match[2]:match[3]], match[1]
	}
	add(speaker, raw[last:])
	return turns
}

func joinTurns(turns []turn) string {
	parts := make([]string, len(turns))
	for i, t := range turns {
		parts[i] = t.text
	}
	return strings.Join(parts, " ")
}

// alignedSegments walks Fish's word list through the turn texts, matching
// letters and digits only (punctuation and cues are not aligned), and cuts a
// segment at every turn and at every pause longer than 800 ms. Each segment's
// text is the turn's own span, so punctuation and emotion cues survive. A word
// that does not match the text where alignment expects it means the two
// disagree, and the caller falls back to plain word grouping without speakers
// rather than attributing speech to the wrong one. When diarize is set, the
// speaker is written onto each aligned word in place.
func alignedSegments(turns []turn, words []batchhttp.Word, diarize bool) ([]runtimepkg.BatchSegment, bool) {
	type position struct{ turn, offset int }
	var stream []rune
	var at []position
	for index, t := range turns {
		masked := cue.ReplaceAllStringFunc(t.text, func(s string) string { return strings.Repeat(" ", len(s)) })
		for offset, r := range masked {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				stream = append(stream, unicode.ToLower(r))
				at = append(at, position{index, offset})
			}
		}
	}
	type placed struct {
		word  *batchhttp.Word
		start position
	}
	placedWords := make([]placed, 0, len(words))
	cursor := 0
	for index := range words {
		word := &words[index]
		first := -1
		for _, r := range strings.ToLower(word.Text) {
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
				continue
			}
			if cursor >= len(stream) || stream[cursor] != r {
				return nil, false
			}
			if first < 0 {
				first = cursor
			}
			cursor++
		}
		if first < 0 {
			continue
		}
		placedWords = append(placedWords, placed{word: word, start: at[first]})
	}
	if len(placedWords) == 0 {
		return nil, false
	}
	var segments []runtimepkg.BatchSegment
	begin := 0
	for i := 1; i <= len(placedWords); i++ {
		if i < len(placedWords) {
			previous, next := placedWords[i-1], placedWords[i]
			if next.start.turn == previous.start.turn && next.word.StartMS-previous.word.EndMS <= 800 {
				continue
			}
		}
		head := placedWords[begin]
		t := turns[head.start.turn]
		from := head.start.offset
		if begin == 0 || placedWords[begin-1].start.turn != head.start.turn {
			from = 0 // a turn's leading cue belongs to its first segment
		}
		to := len(t.text)
		if i < len(placedWords) && placedWords[i].start.turn == head.start.turn {
			to = placedWords[i].start.offset
		}
		segment := runtimepkg.BatchSegment{Text: strings.TrimSpace(t.text[from:to]), StartMS: head.word.StartMS, EndMS: placedWords[i-1].word.EndMS}
		if diarize {
			segment.Speaker = t.speaker
		}
		segments = append(segments, segment)
		begin = i
	}
	if diarize {
		for _, p := range placedWords {
			p.word.Speaker = turns[p.start.turn].speaker
		}
	}
	return segments, true
}

func baseLanguage(language string) string {
	lowered := strings.ToLower(strings.TrimSpace(language))
	if index := strings.IndexAny(lowered, "-_"); index > 0 {
		return lowered[:index]
	}
	return lowered
}
