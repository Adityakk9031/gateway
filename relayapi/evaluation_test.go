package relayapi

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func evaluationRoute() Routing {
	return Routing{Mode: RoutingModeExplicit, Provider: "typesafe", Model: "jev-1.13.0"}
}

func raw(value string) json.RawMessage { return json.RawMessage(value) }

func TestEvaluationMixedStructuredBatch(t *testing.T) {
	t.Parallel()
	request := EvaluationRequest{
		Routing: evaluationRoute(),
		State:   raw(`{"transcript":"こんにちは — I need a person","cart":{"total_cents":1200}}`),
		Questions: map[string]EvaluationQuestion{
			"intent": {
				Type: EvaluationQuestionChoice, Instructions: raw(`{"task":"classify","version":2}`),
				Criteria: raw(`{"order":{"description":"existing order","priority":1},"other":null}`),
			},
			"handoff": {
				Type: EvaluationQuestionNoul, Instructions: raw(`"Was a human explicitly requested?"`),
				Criteria: raw(`{"true":{"must_be":"explicit"},"false":null}`),
			},
			"frustration": {
				Type: EvaluationQuestionScore, Instructions: raw(`["rate","frustration"]`),
				Criteria: raw(`["calm",{"label":"frustrated","severity":1},["angry",2]]`),
			},
		},
	}
	if err := request.Validate(); err != nil {
		t.Fatalf("request invalid: %v", err)
	}
	noul := 0.97
	score := 1.6
	choiceConfidence := 0.92
	scoreConfidence := 0.81
	response := EvaluationResponse{
		Model: "jev-1.13.0",
		Answers: map[string]EvaluationAnswer{
			"intent":  {Type: EvaluationQuestionChoice, Choice: "order", Probabilities: map[string]float64{"order": 0.95, "other": 0.05}, Confidence: &choiceConfidence},
			"handoff": {Type: EvaluationQuestionNoul, Noul: &noul},
			"frustration": {
				Type: EvaluationQuestionScore, Score: &score, Confidence: &scoreConfidence,
				Probabilities: map[string]float64{"0": 0.05, "1": 0.3, "2": 0.65},
				Legend:        map[string]json.RawMessage{"0": raw(`"calm"`), "1": raw(`{"label":"frustrated","severity":1}`), "2": raw(`["angry",2]`)},
			},
		},
		Usage: Usage{InputTokens: 1_000, OutputTokens: 24},
	}
	if err := response.ValidateFor(request); err != nil {
		t.Fatalf("response invalid: %v", err)
	}
}

func TestEvaluationResponseAllowsSemanticallyEquivalentStructuredLegend(t *testing.T) {
	t.Parallel()
	request := EvaluationRequest{
		Routing: evaluationRoute(), State: raw(`{"turn":1}`),
		Questions: map[string]EvaluationQuestion{
			"frustration": {
				Type: EvaluationQuestionScore, Instructions: raw(`"Rate frustration"`),
				Criteria: raw(`[{"label":"calm","severity":0},{"label":"angry","severity":1}]`),
			},
		},
	}
	confidence := 0.91
	score := 1.0
	response := EvaluationResponse{
		Model: "jev-1.13.0",
		Answers: map[string]EvaluationAnswer{
			"frustration": {
				Type: EvaluationQuestionScore, Score: &score, Confidence: &confidence,
				Probabilities: map[string]float64{"0": 0.1, "1": 0.9},
				Legend: map[string]json.RawMessage{
					"0": raw(`{ "severity": 0, "label": "calm" }`),
					"1": raw(`{"severity":1,"label":"angry"}`),
				},
			},
		},
		Usage: Usage{InputTokens: 12},
	}
	if err := response.ValidateFor(request); err != nil {
		t.Fatalf("ValidateFor: %v", err)
	}
}

func TestEvaluationQuestionLimits(t *testing.T) {
	t.Parallel()
	choice := make(map[string]any, MaxChoiceOptions)
	for index := 0; index < MaxChoiceOptions; index++ {
		choice[fmt.Sprintf("option_%03d", index)] = nil
	}
	choiceJSON, _ := json.Marshal(choice)
	if err := (EvaluationQuestion{Type: EvaluationQuestionChoice, Instructions: raw(`"pick"`), Criteria: choiceJSON}).Validate(); err != nil {
		t.Fatalf("255 choices rejected: %v", err)
	}
	choice["overflow"] = nil
	choiceJSON, _ = json.Marshal(choice)
	if err := (EvaluationQuestion{Type: EvaluationQuestionChoice, Instructions: raw(`"pick"`), Criteria: choiceJSON}).Validate(); err == nil {
		t.Fatal("256 choices accepted")
	}
	if err := (EvaluationQuestion{Type: EvaluationQuestionScore, Instructions: raw(`"rate"`), Criteria: raw(`[0,1]`)}).Validate(); err == nil {
		t.Fatal("numeric score legends accepted")
	}
	if err := (EvaluationQuestion{Type: EvaluationQuestionScore, Instructions: raw(`"rate"`), Criteria: raw(`["0","1","2","3","4","5","6","7","8","9","10"]`)}).Validate(); err == nil {
		t.Fatal("11 score levels accepted")
	}
}

func TestEvaluationRejectsUnknownControlsAndAutomaticRouting(t *testing.T) {
	t.Parallel()
	for _, field := range []string{"stream", "temperature", "tools", "max_output_tokens"} {
		body := fmt.Sprintf(`{"routing":{"mode":"explicit","provider":"typesafe","model":"jev-1.13.0"},"state":"x","questions":{"q":{"type":"noul","instructions":"yes?"}},%q:true}`, field)
		var request EvaluationRequest
		if err := json.Unmarshal([]byte(body), &request); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("%s was not rejected as an unknown control: %v", field, err)
		}
	}
	auto := EvaluationRequest{Routing: Routing{Mode: RoutingModeAuto, Objective: ObjectiveBalanced}, State: raw(`"x"`), Questions: map[string]EvaluationQuestion{"q": {Type: EvaluationQuestionNoul, Instructions: raw(`"yes?"`)}}}
	if err := auto.Validate(); err == nil {
		t.Fatal("automatic evaluation route accepted")
	}
}

func TestEvaluationRejectsMalformedAnswers(t *testing.T) {
	t.Parallel()
	request := EvaluationRequest{Routing: evaluationRoute(), State: raw(`"x"`), Questions: map[string]EvaluationQuestion{
		"choice": {Type: EvaluationQuestionChoice, Instructions: raw(`"pick"`), Criteria: raw(`{"a":null,"b":null}`)},
	}}
	confidence := 0.8
	cases := []EvaluationResponse{
		{Model: "jev-1.13.0", Answers: map[string]EvaluationAnswer{}, Usage: Usage{InputTokens: 1}},
		{Model: "jev-1.13.0", Answers: map[string]EvaluationAnswer{"choice": {Type: EvaluationQuestionNoul}}, Usage: Usage{InputTokens: 1}},
		{Model: "jev-1.13.0", Answers: map[string]EvaluationAnswer{"choice": {Type: EvaluationQuestionChoice, Choice: "c", Probabilities: map[string]float64{"a": 0.5, "b": 0.5}, Confidence: &confidence}}, Usage: Usage{InputTokens: 1}},
		{Model: "jev-1.13.0", Answers: map[string]EvaluationAnswer{"choice": {Type: EvaluationQuestionChoice, Choice: "a", Probabilities: map[string]float64{"a": 0.9, "b": 0.2}, Confidence: &confidence}}, Usage: Usage{InputTokens: 1}},
		{Model: "jev-1.13.0", Answers: map[string]EvaluationAnswer{"choice": {Type: EvaluationQuestionChoice, Choice: "a", Probabilities: map[string]float64{"a": 0.9, "b": 0.1}, Confidence: &confidence}}, Usage: Usage{}},
	}
	for index, response := range cases {
		if err := response.ValidateFor(request); err == nil {
			t.Fatalf("malformed case %d accepted", index)
		}
	}
}
