package core

import (
	"bytes"
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestRealtimeNoisyTelephoneFixtureIsDeterministic(t *testing.T) {
	foreground := make([]int16, 2400)
	background := make([]int16, 1200)
	for i := range foreground {
		foreground[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/24000))
	}
	for i := range background {
		background[i] = int16(6000 * math.Sin(2*math.Pi*730*float64(i)/24000))
	}
	first := mixRealtimeTelephoneScene(
		foreground,
		[]realtimeBackgroundTrack{{PCM: background, Gain: 0.2, OffsetMS: 20}},
		0.08,
	)
	second := mixRealtimeTelephoneScene(
		foreground,
		[]realtimeBackgroundTrack{{PCM: background, Gain: 0.2, OffsetMS: 20}},
		0.08,
	)
	if !slices.Equal(first, second) {
		t.Fatal("noisy telephone fixture is not deterministic")
	}
	telephone := telephoneMuLawRoundTrip(first)
	if len(telephone) < len(first)-2 || len(telephone) > len(first)+2 {
		t.Fatalf("telephone round trip changed duration: before=%d after=%d", len(first), len(telephone))
	}
	if slices.Equal(telephone[:min(len(telephone), len(first))], first[:min(len(telephone), len(first))]) {
		t.Fatal("telephone round trip did not alter the source signal")
	}

	recorded := make([]int16, 3600)
	for i := range recorded {
		recorded[i] = int16(5000 * math.Sin(2*math.Pi*310*float64(i)/24000))
	}
	window := realtimeAudioWindow(recorded, 125*time.Millisecond, len(foreground))
	if len(window) != len(foreground) {
		t.Fatalf("real-audio window length = %d, want %d", len(window), len(foreground))
	}
	scaled := scaleRealtimeBackgroundToRatio(foreground, window, 0.4)
	mixed := mixRealtimeTelephoneScene(
		foreground,
		[]realtimeBackgroundTrack{{PCM: scaled, Gain: 1}},
		0,
	)
	ratio := realtimeResidualRMSRatio(foreground, mixed)
	if ratio < 0.39 || ratio > 0.41 {
		t.Fatalf("real-audio background RMS ratio = %.3f, want approximately 0.4", ratio)
	}
}

// TestGoogleRealtimeLiveNoisyTelephone is an opt-in paid Gemini Live
// evaluation. It synthesizes independent foreground/background speakers with
// Gemini, places the callers in a deterministic busy-train-station scene,
// applies an 8 kHz G.711 μ-law telephone round trip, and runs a complete
// multi-turn conversation through Core's production Gemini realtime session
// with the telephony VAD profile.
//
// It also sends loud telephone-shaped noise without speech and verifies that
// Gemini does not create a caller turn.
//
//	RUN_GOOGLE_REALTIME_NOISE_SMOKE=1 GOOGLE_API_KEY=... \
//	  go test -v -run TestGoogleRealtimeLiveNoisyTelephone -timeout 6m .
func TestGoogleRealtimeLiveNoisyTelephone(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_REALTIME_NOISE_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_NOISE_SMOKE=1 to run the paid Gemini noisy-telephone smoke")
	}
	if testing.Short() {
		t.Skip("skipping paid Gemini noisy-telephone smoke in short mode")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	provider := NewGoogleRealtimeProvider(key)

	const (
		callbackTimeText = "Rappelez-moi lundi prochain à seize heures, s'il vous plaît."
		confirmationText = "Oui, c'est bien cela."
	)
	callbackTime := synthesizeGoogleRealtimeSpeech(t, provider, "Aoede", callbackTimeText)
	confirmation := synthesizeGoogleRealtimeSpeech(t, provider, "Aoede", confirmationText)
	const sampleRate = 24000
	sceneName := "train-station"
	var callbackTimeAtStation, confirmationAtStation []int16
	if backgroundPath := strings.TrimSpace(os.Getenv("GOOGLE_REALTIME_BACKGROUND_AUDIO")); backgroundPath != "" {
		sceneName = "real-street"
		background := decodeRealtimeBackgroundAudio(t, backgroundPath)
		firstBackground := realtimeAudioWindow(
			background, 36*time.Second, len(callbackTime)+sampleRate/5,
		)
		firstBackground = scaleRealtimeBackgroundToRatio(callbackTime, firstBackground, 0.42)
		callbackTimeAtStation = mixRealtimeTelephoneScene(
			callbackTime,
			[]realtimeBackgroundTrack{{PCM: fitRealtimeBackground(firstBackground, len(callbackTime)+sampleRate/5), Gain: 1}},
			0,
		)
		secondBackground := realtimeAudioWindow(
			background, 79*time.Second, len(confirmation)+sampleRate/5,
		)
		secondBackground = scaleRealtimeBackgroundToRatio(confirmation, secondBackground, 0.34)
		confirmationAtStation = mixRealtimeTelephoneScene(
			confirmation,
			[]realtimeBackgroundTrack{{PCM: fitRealtimeBackground(secondBackground, len(confirmation)+sampleRate/5), Gain: 1}},
			0,
		)
		t.Logf("using real background audio: %s", backgroundPath)
	} else {
		backgroundFrench := synthesizeGoogleRealtimeSpeech(
			t, provider, "Puck",
			"Le train à destination de Lyon partira voie quatre avec quelques minutes de retard.",
		)
		backgroundEnglish := synthesizeGoogleRealtimeSpeech(
			t, provider, "Kore",
			"Excuse me, is platform seven on the other side of the station?",
		)
		firstDuration := time.Duration(len(callbackTime)+sampleRate/3) * time.Second / sampleRate
		firstFrench := fitRealtimeBackground(
			backgroundFrench,
			len(callbackTime)+sampleRate/5-80*sampleRate/1000,
		)
		firstEnglish := fitRealtimeBackground(
			backgroundEnglish,
			len(callbackTime)+sampleRate/5-620*sampleRate/1000,
		)
		callbackTimeAtStation = mixRealtimeTelephoneScene(
			callbackTime,
			[]realtimeBackgroundTrack{
				{PCM: firstFrench, Gain: 0.23, OffsetMS: 80},
				{PCM: firstEnglish, Gain: 0.16, OffsetMS: 620},
				{PCM: deterministicTrainStationSounds(firstDuration), Gain: 1, OffsetMS: 0},
			},
			0.045,
		)
		secondDuration := time.Duration(len(confirmation)+sampleRate/3) * time.Second / sampleRate
		secondFrench := fitRealtimeBackground(
			backgroundFrench,
			len(confirmation)+sampleRate/5-260*sampleRate/1000,
		)
		secondEnglish := fitRealtimeBackground(
			backgroundEnglish,
			len(confirmation)+sampleRate/5-740*sampleRate/1000,
		)
		confirmationAtStation = mixRealtimeTelephoneScene(
			confirmation,
			[]realtimeBackgroundTrack{
				{PCM: secondFrench, Gain: 0.17, OffsetMS: 260},
				{PCM: secondEnglish, Gain: 0.11, OffsetMS: 740},
				{PCM: deterministicTrainStationSounds(secondDuration), Gain: 0.85, OffsetMS: 0},
			},
			0.035,
		)
	}
	backgroundRatio := realtimeResidualRMSRatio(callbackTime, callbackTimeAtStation)
	t.Logf("%s background-to-foreground RMS ratio before telephone codec: %.2f", sceneName, backgroundRatio)
	if backgroundRatio < 0.35 || backgroundRatio > 0.70 {
		t.Fatalf("%s fixture must remain moderately noisy: RMS ratio %.2f", sceneName, backgroundRatio)
	}
	callbackTimeAtStation = telephoneMuLawRoundTrip(callbackTimeAtStation)
	confirmationAtStation = telephoneMuLawRoundTrip(confirmationAtStation)
	noiseOnly := telephoneMuLawRoundTrip(deterministicRealtimeNoise(3*time.Second, 0.18))
	writeRealtimeNoiseSmokeArtifact(t, "gemini-"+sceneName+"-callback-time.s16le", callbackTimeAtStation)
	writeRealtimeNoiseSmokeArtifact(t, "gemini-"+sceneName+"-confirmation.s16le", confirmationAtStation)
	writeRealtimeNoiseSmokeArtifact(t, "gemini-noise-only.s16le", noiseOnly)

	t.Run("multi-turn-"+sceneName+"-conversation", func(t *testing.T) {
		transcript, fullInteraction := runGoogleRealtimeNoisyConversation(
			t, provider, callbackTimeAtStation, confirmationAtStation,
		)
		t.Logf("GEMINI %s FULL TRANSCRIPT\n%s", strings.ToUpper(sceneName), strings.Join(transcript, "\n"))
		writeRealtimeNoiseSmokeArtifact(t, "gemini-"+sceneName+"-full-interaction.s16le", fullInteraction)
	})

	t.Run("telephone-noise-without-speech", func(t *testing.T) {
		input, output, responses, _ := runGoogleRealtimeAudioRecognition(t, provider, noiseOnly, 5*time.Second)
		t.Logf(
			"GEMINI NOISE-ONLY RESULT\nRESPONSES: %d\nINPUT TRANSCRIPTION: %q\nASSISTANT: %q",
			responses, input, output,
		)
		if responses != 0 || strings.TrimSpace(input) != "" || strings.TrimSpace(output) != "" {
			t.Fatalf("noise created a false caller turn: responses=%d input=%q output=%q", responses, input, output)
		}
	})
}

func writeRealtimeNoiseSmokeArtifact(t *testing.T, name string, pcm []int16) {
	t.Helper()
	dir := strings.TrimSpace(os.Getenv("GOOGLE_REALTIME_NOISE_ARTIFACT_DIR"))
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("create realtime noise artifact directory: %v", err)
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, pcm16SamplesToBytes(pcm), 0o644); err != nil {
		t.Fatalf("write realtime noise artifact %s: %v", name, err)
	}
	t.Logf("wrote realtime noise artifact: %s", path)
}

func decodeRealtimeBackgroundAudio(t *testing.T, path string) []int16 {
	t.Helper()
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal("GOOGLE_REALTIME_BACKGROUND_AUDIO requires ffmpeg in PATH")
	}
	command := exec.Command(
		ffmpeg,
		"-hide_banner", "-loglevel", "error",
		"-i", path,
		"-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000",
		"pipe:1",
	)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		t.Fatalf("decode realtime background audio %s: %v: %s", path, err, strings.TrimSpace(stderr.String()))
	}
	if stdout.Len() < 24000*2 {
		t.Fatalf("decoded realtime background audio is too short: %s (%d PCM bytes)", path, stdout.Len())
	}
	return bytesToPCM16Samples(stdout.Bytes())
}

func realtimeAudioWindow(source []int16, start time.Duration, samples int) []int16 {
	if len(source) == 0 || samples <= 0 {
		return nil
	}
	startSample := int(start * 24000 / time.Second)
	startSample %= len(source)
	out := make([]int16, samples)
	for i := range out {
		out[i] = source[(startSample+i)%len(source)]
	}
	return out
}

func scaleRealtimeBackgroundToRatio(foreground, background []int16, target float64) []int16 {
	count := min(len(foreground), len(background))
	if count == 0 || target <= 0 {
		return make([]int16, len(background))
	}
	var foregroundEnergy, backgroundEnergy float64
	for i := range count {
		foregroundSample := float64(foreground[i])
		backgroundSample := float64(background[i])
		foregroundEnergy += foregroundSample * foregroundSample
		backgroundEnergy += backgroundSample * backgroundSample
	}
	if foregroundEnergy == 0 || backgroundEnergy == 0 {
		return make([]int16, len(background))
	}
	gain := target * math.Sqrt(foregroundEnergy/backgroundEnergy)
	out := make([]int16, len(background))
	for i, sample := range background {
		out[i] = clampRealtimePCM(float64(sample) * gain)
	}
	return out
}

func synthesizeGoogleRealtimeSpeech(t *testing.T, provider *GoogleRealtimeProvider, voice, text string) []int16 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: voice,
		Instructions: `Speak the requested sentence exactly once, naturally and clearly.
Do not introduce it, comment on it, translate it, or add any other words.`,
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
	})
	if err != nil {
		t.Fatalf("open Gemini speech synthesizer %s: %v", voice, err)
	}
	defer session.Close()
	if err := session.RestoreConversation(nil); err != nil {
		t.Fatalf("restore Gemini speech synthesizer %s: %v", voice, err)
	}
	if err := session.SendText("user", "Say exactly: "+text); err != nil {
		t.Fatalf("send Gemini speech synthesizer %s: %v", voice, err)
	}

	var pcm []byte
	var transcript string
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("Gemini speech synthesizer %s timed out; transcript=%q bytes=%d", voice, transcript, len(pcm))
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("Gemini speech synthesizer %s ended early", voice)
			}
			switch event.Type {
			case RealtimeEventError:
				t.Fatalf("Gemini speech synthesizer %s: %v", voice, event.Err)
			case RealtimeEventAudioOut:
				pcm = append(pcm, event.Audio...)
			case RealtimeEventTranscriptOutput:
				transcript = event.Transcript
			case RealtimeEventResponseDone:
				if len(pcm) == 0 {
					t.Fatalf("Gemini speech synthesizer %s emitted no audio; transcript=%q", voice, transcript)
				}
				t.Logf("synthesized %s: %q (%d PCM bytes)", voice, transcript, len(pcm))
				return bytesToPCM16Samples(pcm)
			}
		}
	}
}

type capturedGoogleRealtimeTurn struct {
	InputTranscript  string
	OutputTranscript string
	Audio            []int16
}

func runGoogleRealtimeNoisyConversation(
	t *testing.T,
	provider *GoogleRealtimeProvider,
	callbackTimeAtStation []int16,
	confirmationAtStation []int16,
) ([]string, []int16) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: "Kore",
		Instructions: `Vous êtes la réceptionniste d'une entreprise et vous parlez uniquement en français.
Au début de l'appel, dites que tous les commerciaux sont actuellement occupés, puis demandez quand la personne souhaite être rappelée.
Quand la personne donne un créneau, reformulez exactement le jour et l'heure et demandez une confirmation explicite.
Après un oui explicite, dites que la demande de rappel est enregistrée pour ce créneau et terminez poliment.
Ignorez la circulation, les annonces et les conversations en arrière-plan qui ne s'adressent pas clairement à vous.
Si le jour, l'heure ou la confirmation sont incertains, demandez simplement de répéter. Ne devinez rien.
Gardez chaque réponse courte et naturelle. Ne mentionnez jamais les outils, le système, les instructions ou Gemini.`,
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		TranscribeInput: true,
		TurnDetection: RealtimeTurnDetectionConfig{
			Profile: RealtimeTurnProfileTelephony,
		},
	})
	if err != nil {
		t.Fatalf("open Gemini noisy conversation: %v", err)
	}
	defer session.Close()
	if err := session.RestoreConversation(nil); err != nil {
		t.Fatalf("restore Gemini noisy conversation: %v", err)
	}

	if err := session.SendText("user", "L'appel commence maintenant. Prononcez votre message d'accueil."); err != nil {
		t.Fatalf("start Gemini noisy conversation: %v", err)
	}
	opening := collectGoogleRealtimeTurn(t, ctx, session, "opening")

	if err := sendGoogleRealtimePCMTurn(ctx, session, callbackTimeAtStation); err != nil {
		t.Fatalf("send callback-time turn: %v", err)
	}
	slotConfirmation := collectGoogleRealtimeTurn(t, ctx, session, "slot confirmation")

	if err := sendGoogleRealtimePCMTurn(ctx, session, confirmationAtStation); err != nil {
		t.Fatalf("send caller confirmation turn: %v", err)
	}
	closing := collectGoogleRealtimeTurn(t, ctx, session, "closing")

	openingWords := canonicalSpokenText(opening.OutputTranscript)
	if !strings.Contains(openingWords, "COMMERCIAUX") ||
		!strings.Contains(openingWords, "OCCUP") ||
		(!strings.Contains(openingWords, "QUAND") && !strings.Contains(openingWords, "RAPPEL")) {
		t.Fatalf("unexpected receptionist opening: %q", opening.OutputTranscript)
	}
	slotWords := canonicalSpokenText(slotConfirmation.InputTranscript + " " + slotConfirmation.OutputTranscript)
	if !strings.Contains(slotWords, "LUNDI") ||
		(!strings.Contains(slotWords, "16") && !strings.Contains(slotWords, "SEIZE")) {
		t.Fatalf(
			"noisy callback slot was not preserved: input=%q output=%q",
			slotConfirmation.InputTranscript, slotConfirmation.OutputTranscript,
		)
	}
	callerConfirmation := canonicalSpokenText(closing.InputTranscript)
	if !strings.Contains(callerConfirmation, "OUI") {
		t.Fatalf("caller confirmation was not understood: %q", closing.InputTranscript)
	}
	closingWords := canonicalSpokenText(closing.OutputTranscript)
	if !strings.Contains(closingWords, "RAPPEL") ||
		(!strings.Contains(closingWords, "ENREGISTR") && !strings.Contains(closingWords, "NOTE")) {
		t.Fatalf("unexpected receptionist closing: %q", closing.OutputTranscript)
	}

	transcript := []string{
		"ASSISTANT: " + opening.OutputTranscript,
		"CALLER: " + slotConfirmation.InputTranscript,
		"ASSISTANT: " + slotConfirmation.OutputTranscript,
		"CALLER: " + closing.InputTranscript,
		"ASSISTANT: " + closing.OutputTranscript,
	}
	var interaction []int16
	appendTurn := func(audio []int16) {
		if len(interaction) > 0 {
			interaction = append(interaction, make([]int16, 24000/2)...)
		}
		interaction = append(interaction, audio...)
	}
	appendTurn(opening.Audio)
	appendTurn(callbackTimeAtStation)
	appendTurn(slotConfirmation.Audio)
	appendTurn(confirmationAtStation)
	appendTurn(closing.Audio)
	return transcript, interaction
}

func collectGoogleRealtimeTurn(
	t *testing.T,
	ctx context.Context,
	session RealtimeSession,
	stage string,
) capturedGoogleRealtimeTurn {
	t.Helper()
	timer := time.NewTimer(25 * time.Second)
	defer timer.Stop()
	var turn capturedGoogleRealtimeTurn
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("Gemini noisy conversation %s timed out: %v", stage, ctx.Err())
		case <-timer.C:
			t.Fatalf(
				"Gemini noisy conversation %s timed out; input=%q output=%q audio_samples=%d",
				stage, turn.InputTranscript, turn.OutputTranscript, len(turn.Audio),
			)
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("Gemini noisy conversation ended during %s", stage)
			}
			switch event.Type {
			case RealtimeEventError:
				t.Fatalf("Gemini noisy conversation %s: %v", stage, event.Err)
			case RealtimeEventAudioOut:
				turn.Audio = append(turn.Audio, bytesToPCM16Samples(event.Audio)...)
			case RealtimeEventTranscriptInput:
				turn.InputTranscript = event.Transcript
			case RealtimeEventTranscriptOutput:
				turn.OutputTranscript = event.Transcript
			case RealtimeEventResponseDone:
				if len(turn.Audio) == 0 || strings.TrimSpace(turn.OutputTranscript) == "" {
					t.Fatalf(
						"Gemini noisy conversation %s returned an incomplete spoken turn: %#v",
						stage, turn,
					)
				}
				return turn
			}
		}
	}
}

func sendGoogleRealtimePCMTurn(ctx context.Context, session RealtimeSession, pcm []int16) error {
	const (
		sampleRate   = 24000
		frameMS      = 20
		frameSamples = sampleRate * frameMS / 1000
	)
	sendFrame := func(frame []byte) error {
		if err := session.SendAudio(frame); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(frameMS * time.Millisecond):
			return nil
		}
	}
	for offset := 0; offset < len(pcm); offset += frameSamples {
		end := min(offset+frameSamples, len(pcm))
		if err := sendFrame(pcm16SamplesToBytes(pcm[offset:end])); err != nil {
			return err
		}
	}
	silence := make([]byte, frameSamples*2)
	for range 75 {
		if err := sendFrame(silence); err != nil {
			return err
		}
	}
	return nil
}

func runGoogleRealtimeAudioRecognition(
	t *testing.T,
	provider *GoogleRealtimeProvider,
	pcm []int16,
	observation time.Duration,
) (input, output string, responses int, assistantAudio []int16) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: "Kore",
		Instructions: `Vous gérez un appel téléphonique en français.
Ignorez les bruits de fond et les conversations qui ne s'adressent pas clairement à vous.
Si une demande complète et cohérente est audible, reformulez seulement le jour et l'heure demandés.
Si l'audio est partiel, chevauché ou incertain, demandez brièvement à la personne de répéter.
Ne devinez jamais un jour ou une heure.`,
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		TranscribeInput: true,
		TurnDetection: RealtimeTurnDetectionConfig{
			Profile: RealtimeTurnProfileTelephony,
		},
	})
	if err != nil {
		t.Fatalf("open Gemini noisy-telephone recognizer: %v", err)
	}
	defer session.Close()
	if err := session.RestoreConversation(nil); err != nil {
		t.Fatalf("restore Gemini noisy-telephone recognizer: %v", err)
	}

	if err := sendGoogleRealtimePCM(ctx, session, pcm); err != nil {
		t.Fatalf("send Gemini noisy-telephone audio: %v", err)
	}
	deadline := time.NewTimer(observation)
	defer deadline.Stop()
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("Gemini noisy-telephone recognizer timed out: %v", ctx.Err())
		case <-deadline.C:
			return input, output, responses, assistantAudio
		case event, ok := <-session.Events():
			if !ok {
				return input, output, responses, assistantAudio
			}
			switch event.Type {
			case RealtimeEventError:
				t.Fatalf("Gemini noisy-telephone recognizer: %v", event.Err)
			case RealtimeEventAudioOut:
				assistantAudio = append(assistantAudio, bytesToPCM16Samples(event.Audio)...)
			case RealtimeEventTranscriptInput:
				input = event.Transcript
			case RealtimeEventTranscriptOutput:
				output = event.Transcript
			case RealtimeEventResponseDone:
				responses++
				// Noise-only must remain observable for the entire window.
				// A real caller response can return as soon as the completed
				// transcript has arrived.
				if strings.TrimSpace(input) != "" || strings.TrimSpace(output) != "" {
					return input, output, responses, assistantAudio
				}
			}
		}
	}
}

func sendGoogleRealtimePCM(ctx context.Context, session RealtimeSession, pcm []int16) error {
	const (
		sampleRate   = 24000
		frameMS      = 20
		frameSamples = sampleRate * frameMS / 1000
	)
	for offset := 0; offset < len(pcm); offset += frameSamples {
		end := min(offset+frameSamples, len(pcm))
		if err := session.SendAudio(pcm16SamplesToBytes(pcm[offset:end])); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(frameMS * time.Millisecond):
		}
	}
	silence := make([]byte, frameSamples*2)
	for range 100 {
		if err := session.SendAudio(silence); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(frameMS * time.Millisecond):
		}
	}
	// This live test has a finite recording rather than an always-open phone
	// stream, so explicitly flush Gemini's cached tail after the silence.
	if google, ok := session.(*googleRealtimeSession); ok {
		return google.enqueue(map[string]any{"realtimeInput": map[string]any{"audioStreamEnd": true}})
	}
	return nil
}

type realtimeBackgroundTrack struct {
	PCM      []int16
	Gain     float64
	OffsetMS int
}

func fitRealtimeBackground(pcm []int16, maxSamples int) []int16 {
	if maxSamples <= 0 || len(pcm) == 0 {
		return nil
	}
	out := slices.Clone(pcm[:min(len(pcm), maxSamples)])
	fadeSamples := min(len(out), 2400)
	for i := range fadeSamples {
		index := len(out) - fadeSamples + i
		out[index] = clampRealtimePCM(float64(out[index]) * float64(fadeSamples-i) / float64(fadeSamples))
	}
	return out
}

func mixRealtimeTelephoneScene(
	foreground []int16,
	background []realtimeBackgroundTrack,
	noiseGain float64,
) []int16 {
	const sampleRate = 24000
	length := len(foreground)
	for _, track := range background {
		end := track.OffsetMS*sampleRate/1000 + len(track.PCM)
		if end > length {
			length = end
		}
	}
	out := make([]int16, length)
	noise := deterministicRealtimeNoise(time.Duration(length)*time.Second/sampleRate, noiseGain)
	var duckedNoise float64
	for i := range out {
		value := 0.0
		if i < len(foreground) {
			value += float64(foreground[i])
		}
		for _, track := range background {
			index := i - track.OffsetMS*sampleRate/1000
			if index >= 0 && index < len(track.PCM) {
				value += float64(track.PCM[index]) * track.Gain
			}
		}
		if i < len(noise) {
			// Slowly varying colored noise is more telephone-like than
			// independent full-band samples.
			duckedNoise = 0.92*duckedNoise + 0.08*float64(noise[i])
			value += duckedNoise
		}
		out[i] = clampRealtimePCM(value)
	}
	return out
}

func deterministicRealtimeNoise(duration time.Duration, gain float64) []int16 {
	const sampleRate = 24000
	count := int(duration * sampleRate / time.Second)
	out := make([]int16, count)
	state := uint32(0x6d2b79f5)
	colored := 0.0
	for i := range out {
		state = 1664525*state + 1013904223
		white := (float64(state>>8)/float64(1<<24))*2 - 1
		colored = 0.88*colored + 0.12*white
		out[i] = clampRealtimePCM(colored * gain * math.MaxInt16)
	}
	return out
}

func deterministicTrainStationSounds(duration time.Duration) []int16 {
	const sampleRate = 24000
	out := deterministicRealtimeNoise(duration, 0.065)
	footstepsAt := []float64{0.42, 1.18, 1.83, 2.76, 3.48}
	for i := range out {
		seconds := float64(i) / sampleRate
		value := float64(out[i])
		// Ventilation and the low rolling sound of a station concourse.
		value += 0.024 * math.MaxInt16 * math.Sin(2*math.Pi*170*seconds)
		value += 0.016 * math.MaxInt16 * math.Sin(2*math.Pi*255*seconds)
		// A short public-address chime before a distant announcement.
		if (seconds >= 0.62 && seconds < 0.82) || (seconds >= 0.91 && seconds < 1.11) {
			value += 0.085 * math.MaxInt16 *
				(math.Sin(2*math.Pi*660*seconds) + 0.45*math.Sin(2*math.Pi*880*seconds))
		}
		// Shoes and suitcase wheels crossing the tiled floor.
		for _, start := range footstepsAt {
			elapsed := seconds - start
			if elapsed >= 0 && elapsed < 0.11 {
				envelope := math.Exp(-28 * elapsed)
				value += 0.09 * math.MaxInt16 * envelope *
					(math.Sin(2*math.Pi*420*elapsed) + 0.35*math.Sin(2*math.Pi*930*elapsed))
			}
		}
		// A train rolls past in the distance without drowning out speech.
		if seconds >= 2.20 && seconds < 3.55 {
			elapsed := seconds - 2.20
			envelope := math.Sin(math.Pi * elapsed / 1.35)
			value += 0.045 * math.MaxInt16 * envelope *
				(math.Sin(2*math.Pi*310*seconds) + 0.4*math.Sin(2*math.Pi*470*seconds))
		}
		out[i] = clampRealtimePCM(value)
	}
	return out
}

func realtimeResidualRMSRatio(foreground, mixed []int16) float64 {
	count := min(len(foreground), len(mixed))
	if count == 0 {
		return 0
	}
	var foregroundEnergy, residualEnergy float64
	for i := range count {
		source := float64(foreground[i])
		residual := float64(mixed[i]) - source
		foregroundEnergy += source * source
		residualEnergy += residual * residual
	}
	if foregroundEnergy == 0 {
		return 0
	}
	return math.Sqrt(residualEnergy / foregroundEnergy)
}

func telephoneMuLawRoundTrip(input []int16) []int16 {
	if len(input) == 0 {
		return nil
	}
	downsampled := make([]int16, 0, (len(input)+2)/3)
	for i := 0; i < len(input); i += 3 {
		end := min(i+3, len(input))
		sum := 0
		for _, sample := range input[i:end] {
			sum += int(sample)
		}
		downsampled = append(downsampled, int16(sum/(end-i)))
	}
	decoded := make([]int16, len(downsampled))
	for i, sample := range downsampled {
		decoded[i] = realtimeMuLawToLinear(realtimeLinearToMuLaw(sample))
	}
	out := make([]int16, 0, len(decoded)*3)
	for _, sample := range decoded {
		out = append(out, sample, sample, sample)
	}
	return out
}

func realtimeLinearToMuLaw(sample int16) byte {
	const (
		bias = 0x84
		clip = 32635
	)
	value := int(sample)
	sign := 0
	if value < 0 {
		sign = 0x80
		value = -value
	}
	if value > clip {
		value = clip
	}
	value += bias
	exponent := 7
	for mask := 0x4000; exponent > 0 && value&mask == 0; exponent-- {
		mask >>= 1
	}
	mantissa := (value >> (exponent + 3)) & 0x0f
	return ^byte(sign | exponent<<4 | mantissa)
}

func realtimeMuLawToLinear(value byte) int16 {
	value = ^value
	sign := value & 0x80
	exponent := (value >> 4) & 0x07
	mantissa := value & 0x0f
	sample := ((int(mantissa) << 3) + 0x84) << exponent
	sample -= 0x84
	if sign != 0 {
		sample = -sample
	}
	return int16(sample)
}

func bytesToPCM16Samples(data []byte) []int16 {
	samples := make([]int16, len(data)/2)
	for i := range samples {
		samples[i] = int16(uint16(data[i*2]) | uint16(data[i*2+1])<<8)
	}
	return samples
}

func pcm16SamplesToBytes(samples []int16) []byte {
	data := make([]byte, len(samples)*2)
	for i, sample := range samples {
		data[i*2] = byte(sample)
		data[i*2+1] = byte(uint16(sample) >> 8)
	}
	return data
}

func clampRealtimePCM(value float64) int16 {
	if value > math.MaxInt16 {
		return math.MaxInt16
	}
	if value < math.MinInt16 {
		return math.MinInt16
	}
	return int16(math.Round(value))
}
