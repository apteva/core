#!/usr/bin/env python3
"""Exercise existing Telephony handlers against current Core using Go overlays.

Does not edit Server/Telephony, change deployed agents, or place PSTN calls.
OPENAI_API_KEY is required only with --live. Reports retain full timing events.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--telephony', type=Path, required=True)
parser.add_argument('--server', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--live', action='store_true')
parser.add_argument('--turns', type=int, default=2, choices=range(1, 4))
parser.add_argument('--pause-ms', type=int, default=0)
a = parser.parse_args()
if not 0 <= a.pause_ms <= 2000:
    parser.error('--pause-ms must be 0..2000')
if a.live and not os.environ.get('OPENAI_API_KEY'):
    parser.error('OPENAI_API_KEY required for --live')
core = Path(__file__).resolve().parent.parent
phone, server, out = a.telephony.resolve(), a.server.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, GOWORK='off', TELEMETRY_URL='', TELEMETRY_LIVE_URL='', SERVER_URL='',
           TELEPHONY_AGENT_AUDIO_LATENCY='1', TELEPHONY_AGENT_AUDIO_LIVE='1' if a.live else '0',
           TELEPHONY_LATENCY_PAUSE_MS=str(a.pause_ms), TELEPHONY_INTERNAL_OUTPUT=str(out))

def run(cmd, cwd, log):
    result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.decode(errors='replace')
    key = env.get('OPENAI_API_KEY')
    if key:
        text = text.replace(key, '[REDACTED]')
    (out / log).write_text(text)
    return result.returncode

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Replay fixture changed; review overlay anchor: ' + old)
    return text.replace(old, new, 1)

with tempfile.TemporaryDirectory(prefix='core-transport-replay-') as temporary:
    tmp = Path(temporary)
    metadata = {'live': a.live, 'turns': a.turns, 'pause_ms': a.pause_ms, 'sources': {}}
    for component, root, fixture in [
        ('core', core, phone / 'benchmarks/agent-audio/core_fixture.go.txt'),
        ('server', server, phone / 'benchmarks/internal-audio/server_fixture.go.txt')]:
        if component == 'core':
            fixture_text = fixture.read_text()
            fixture_text = replace_once(fixture_text, ' mux.HandleFunc("/emit",', '\n mux.HandleFunc("/start-turn",func(w http.ResponseWriter,r *http.Request){capture.mu.Lock();capture.replies=0;capture.bytes.Store(0);capture.mu.Unlock();w.WriteHeader(204)})\n mux.HandleFunc("/emit",')
            fixture = tmp / 'core_fixture.go'
            fixture.write_text(fixture_text)
        overlay = tmp / (component + '.json')
        virtual = root / 'zz_core_transport_replay_test.go'
        if virtual.exists():
            raise RuntimeError('Overlay would replace existing file')
        overlay.write_text(json.dumps({'Replace': {str(virtual): str(fixture)}}))
        binary = tmp / component
        if run(['go', 'test', '-c', '-overlay', str(overlay), '-o', str(binary), '.'], root, component + '-build.log'):
            raise SystemExit(component + ' build failed; see ' + str(out))
        env['TELEPHONY_INTERNAL_' + component.upper() + '_HELPER'] = str(binary)
        metadata['sources'][component] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob('*.go')}
    source = phone / 'agent_latency_integration_test.go'
    text = source.read_text()
    # Keep one provider session and carrier bridge across turns. A function per
    # turn scopes microphone cleanup without overlapping two input producers.
    anchor = '\t\t\t\tpcm := newPCMResampler(24000, probe.sampleRate).Process(speech)'
    text = replace_once(text, anchor, f'\t\t\t\tfor turn := 1; turn <= {a.turns}; turn++ {{\n\t\t\t\tfunc() {{\n\t\t\t\tagentFixtureJSON(t, \"POST\", core.url+\"/start-turn\", nil, nil)\n\t\t\t\tturnStart := time.Now()\n' + anchor)
    text = replace_once(text, '\n\t\t\t})\n\t\t}', '\n\t\t\t\t}()\n\t\t\t\t}\n\t\t\t})\n\t\t}')
    text = replace_once(text, 'if e.Type == "realtime.assistant" &&', 'if e.Time.After(turnStart) && e.Type == "realtime.assistant" &&')
    text = replace_once(text, 'filepath.Join(output, "latency.json")', 'filepath.Join(output, fmt.Sprintf("latency-turn-%d.json", turn))')
    # A 400ms stall necessarily exceeds a 150ms peak threshold. Retain that
    # peak in reports; independently assert that the last two flow windows
    # recover below 150ms. Healthy runs retain the original peak assertion.
    text = replace_once(text, 'if maxInput > 150 {', 'if pauseMS == 0 && maxInput > 150 {')
    anchor = '\t\t\t\tif live && (first.Sub(ended)'
    check = '''
                if pauseMS > 0 {
                    var waits []float64
                    for _, e := range stats.Events {
                        if e.Type == "realtime.input_flow" {
                            if n, ok := e.Data["input_wait_max_ms"].(float64); ok { waits = append(waits,n) }
                        }
                    }
                    if len(waits)<3 { t.Fatal("insufficient post-stall evidence") }
                    for _, n := range waits[len(waits)-2:] {
                        if n>150 { t.Errorf("persistent post-stall input wait: %.1fms", n) }
                    }
                }
'''
    text = replace_once(text, anchor, check + anchor)
    patched = out / 'latency_replay_test.go.txt'
    patched.write_text(text)
    metadata['latency_fixture_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / 'sources.json').write_text(json.dumps(metadata, indent=2))
    overlay = tmp / 'telephony.json'
    overlay.write_text(json.dumps({'Replace': {str(source): str(patched)}}))
    code = run(['go', 'test', '-overlay', str(overlay), '-tags', 'integration', '-run', '^TestAgentAudioMatchedLatency$', '-count=1', '-timeout', '6m', '-v', '.'], phone, 'verification.log')
    print('Replay exit:', code, 'Reports:', out, flush=True)
    raise SystemExit(code)
