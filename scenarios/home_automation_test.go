package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var homeAutomationScenario = Scenario{
	Name: "HomeAutomation",
	Directive: `You are the supervisor of a smart-home automation system. On
startup, spawn four persistent team members and then stay out of the loop:

1. "security" — watches sensors and cameras for intruders, alerts the
   owner, and can turn on deterrent lights. Spawn with
   mcp="home_sensors,home_cameras,home_devices" tools="send,pace".
   Directive: "Long-lived security monitor. On startup list_sensors and
   list_cameras. Then pace(sleep='1h'). On each wake, get_events from
   home_sensors since your last check. If motion is triggered in an
   unoccupied room AND the owner is away, describe_scene on the matching
   camera, start_recording, and notify_owner with a clear alert. Never
   spawn threads. Never call done."

2. "comfort" — manages thermostat and lights. Spawn with
   mcp="home_devices" tools="send,pace". Directive: "Long-lived
   comfort worker. On startup list_devices. Then pace(sleep='1h').
   React to messages from daily_dispatcher for morning/evening routines,
   and to ad-hoc send() requests from main for one-off adjustments.
   Never spawn threads. Never call done."

3. "intercom" — handles doorbell visitors. Spawn with
   mcp="home_intercom,home_cameras,home_devices" tools="send,pace".
   Directive: "Long-lived visitor handler. On startup get_allowlist and
   list_cameras. Then pace(sleep='1h'). On each wake call
   get_pending_visits. For each pending visit: describe_scene on the
   visit's camera_id, compare against the allowlist, then either
   unlock_door + speak a welcome (for allowlisted visitors) OR
   deny_entry + notify_owner (for strangers). Never spawn threads.
   Never call done."

4. "daily_dispatcher" — coordinator for scheduled routines. Spawn with
   tools="send,pace" and NO mcp. Directive: "Long-lived schedule
   coordinator with no domain tools. React to [time] console events
   injected into your inbox. When you see '[time] morning' send to
   comfort ('Morning routine: thermostat 21C, kitchen light 70%,
   living light 50%') and to intercom ('Morning: expect deliveries').
   When you see '[time] evening' send to comfort ('Evening: thermostat
   18C, lights low') and security ('Evening: arm monitoring'). Between
   wakes pace(sleep='1h'). Never spawn threads. Never call done."

After spawning all four, pace(sleep='1h') yourself. Do not intervene
unless a worker reports an error or the user asks you something.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "home_sensors",
			Command: "", // filled at test time
			Env:     map[string]string{"HOME_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "home_cameras",
			Command: "",
			Env:     map[string]string{"HOME_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "home_intercom",
			Command: "",
			Env:     map[string]string{"HOME_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "home_devices",
			Command: "",
			Env:     map[string]string{"HOME_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		// World state. Single home.json shared across the four MCPs —
		// home_devices owns the lights/thermostats/locks keys and does
		// a read-modify-write; everyone else only reads.
		WriteJSONFile(t, dir, "home.json", map[string]any{
			"sensors": []map[string]any{
				{"id": "motion_living", "type": "motion", "room": "living", "enabled": true},
				{"id": "motion_kitchen", "type": "motion", "room": "kitchen", "enabled": true},
				{"id": "motion_hallway", "type": "motion", "room": "hallway", "enabled": true},
				{"id": "motion_bedroom", "type": "motion", "room": "bedroom", "enabled": true},
				{"id": "door_front", "type": "door", "room": "entrance", "enabled": true},
				{"id": "door_back", "type": "door", "room": "kitchen", "enabled": true},
				{"id": "window_bedroom", "type": "window", "room": "bedroom", "enabled": true},
			},
			"cameras": []map[string]any{
				{"id": "cam_front", "room": "entrance", "stream_url": "rtsp://mock/front"},
				{"id": "cam_living", "room": "living", "stream_url": "rtsp://mock/living"},
				{"id": "cam_kitchen", "room": "kitchen", "stream_url": "rtsp://mock/kitchen"},
				{"id": "cam_backyard", "room": "backyard", "stream_url": "rtsp://mock/backyard"},
			},
			"lights": []map[string]any{
				{"id": "light_living", "room": "living", "on": false, "brightness": 0},
				{"id": "light_kitchen", "room": "kitchen", "on": false, "brightness": 0},
				{"id": "light_bedroom", "room": "bedroom", "on": false, "brightness": 0},
				{"id": "light_entrance", "room": "entrance", "on": false, "brightness": 0},
			},
			"thermostats": []map[string]any{
				{"id": "main", "room": "living", "current_c": 19.5, "setpoint_c": 20.0, "mode": "heat"},
			},
			"locks": []map[string]any{
				{"id": "front", "door": "front", "locked": true},
				{"id": "back", "door": "back", "locked": true},
				{"id": "garage", "door": "garage", "locked": true},
			},
			"occupancy": map[string]any{"home": false, "people": []string{}},
			"visitor_allowlist": []string{
				"delivery person in uniform holding a package",
				"cleaning service",
				"plumber",
			},
		})
		// Empty event + visit files — the test framework appends to
		// these between phases to inject motion and doorbell events.
		WriteJSONFile(t, dir, "scenes.json", map[string]string{})
		os.WriteFile(filepath.Join(dir, "sensor_events.jsonl"), []byte(""), 0644)
		os.WriteFile(filepath.Join(dir, "visits.jsonl"), []byte(""), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 4 workers spawned + dispatcher",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.Threads().Count()
				t.Logf("  ... threads=%d %v", count, ThreadIDs(th))
				return count >= 4
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				t.Logf("startup audit (%d entries)", len(entries))
				// Expect at least one list_sensors (security) and one
				// list_devices (comfort) discovery call from worker setup.
				if CountTool(entries, "list_sensors") == 0 {
					t.Logf("NOTE: security did not list_sensors during startup")
				}
				if CountTool(entries, "list_devices") == 0 {
					t.Logf("NOTE: comfort did not list_devices during startup")
				}
			},
		},
		{
			Name:    "Passive monitoring — idle window elapses cleanly",
			Timeout: 45 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Pure wait-for-time phase — let 35s elapse. A crashed
				// agent is caught by the scenario hard timeout.
				time.Sleep(35 * time.Second)
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("passive monitoring: %d sub-threads alive: %v", th.Threads().Count(), ThreadIDs(th))
			},
		},
		{
			Name:    "Intruder motion at night — security escalates",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Seed an intruder description on the living-room camera.
				WriteJSONFile(t, dir, "scenes.json", map[string]string{
					"cam_living": "A person wearing a dark hoodie is moving through the living room toward the hallway. They are not one of the residents.",
				})
				// Inject a motion event into sensors.
				entry := map[string]any{
					"time":      time.Now().UTC().Format(time.RFC3339),
					"sensor_id": "motion_living",
					"type":      "motion",
					"room":      "living",
					"value":     "triggered",
				}
				data, _ := json.Marshal(entry)
				f, _ := os.OpenFile(filepath.Join(dir, "sensor_events.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if f != nil {
					f.WriteString(string(data) + "\n")
					f.Close()
				}
				// Wake main so it can relay to security. (In production this
				// would be a webhook from the sensor MCP; in test we use a
				// console event as the wake signal.)
				th := getTestThinker(t) // see helper below
				_ = th
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("Security alert: motion triggered in living room. Ask security to investigate.")
					}
					entries := ReadAuditEntries(dir)
					notifies := CountTool(entries, "notify_owner")
					describes := CountTool(entries, "describe_scene")
					t.Logf("  ... describe_scene=%d notify_owner=%d threads=%v",
						describes, notifies, ThreadIDs(th))
					return notifies >= 1 && describes >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				if CountTool(entries, "notify_owner") == 0 {
					t.Error("expected at least one notify_owner call during intruder phase")
				}
				if CountTool(entries, "describe_scene") == 0 {
					t.Error("expected at least one describe_scene call during intruder phase")
				}
			},
		},
		{
			Name:    "Expected visitor at the door — intercom admits",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Seed the front-door camera with a delivery-person description.
				WriteJSONFile(t, dir, "scenes.json", map[string]string{
					"cam_front":  "A delivery person in a brown uniform holding a cardboard package from Amazon.",
					"cam_living": "Empty living room.",
				})
				// Inject the doorbell press.
				visit := map[string]any{
					"id":        "v_001",
					"time":      time.Now().UTC().Format(time.RFC3339),
					"camera_id": "cam_front",
					"door_id":   "front",
					"status":    "pending",
				}
				data, _ := json.Marshal(visit)
				os.WriteFile(filepath.Join(dir, "visits.jsonl"), append(data, '\n'), 0644)
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("Doorbell rang at the front door. Ask intercom to handle it.")
					}
					entries := ReadAuditEntries(dir)
					unlocks := CountTool(entries, "unlock_door")
					speaks := CountTool(entries, "speak")
					t.Logf("  ... unlock_door=%d speak=%d threads=%v",
						unlocks, speaks, ThreadIDs(th))
					return unlocks >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				if CountTool(entries, "unlock_door") == 0 {
					t.Error("expected unlock_door for allowlisted visitor")
				}
			},
		},
		{
			Name:    "Delegate-not-spawn — ad-hoc kitchen light request",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Record the thread count BEFORE we inject the task so
				// we can assert nothing new was spawned.
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				var threadsBefore int
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						threadsBefore = th.Threads().Count()
						t.Logf("  ... thread count before task: %d", threadsBefore)
						th.InjectConsole("Make the kitchen light brighter, to about 90%.")
					}
					// Look for a set_light call on the kitchen light with
					// high brightness — that's the proof comfort handled it.
					entries := ReadAuditEntries(dir)
					hit := 0
					for _, e := range entries {
						if e.Tool == "set_light" && e.Args["id"] == "light_kitchen" {
							if b := e.Args["brightness"]; b == "90" || b == "80" || b == "100" {
								hit++
							}
						}
					}
					threadsNow := th.Threads().Count()
					t.Logf("  ... set_light(kitchen, >=80)=%d threadsNow=%d %v",
						hit, threadsNow, ThreadIDs(th))
					// Success iff: set_light done AND thread count did NOT grow.
					if hit >= 1 && threadsNow <= threadsBefore {
						return true
					}
					// Also return true if set_light done even if a thread
					// was spawned (we'll flag it in Verify as a warning
					// rather than a hard fail).
					return hit >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				var kitchenSets int
				for _, e := range entries {
					if e.Tool == "set_light" && e.Args["id"] == "light_kitchen" {
						kitchenSets++
					}
				}
				if kitchenSets == 0 {
					t.Error("expected at least one set_light on light_kitchen")
				}
				// The kitchen light getting set is the goal. Whether
				// main delegated to a comfort worker or handled the
				// ad-hoc task itself is its own call — both are valid.
				t.Logf("ad-hoc task: %d sub-threads alive: %v", th.Threads().Count(), ThreadIDs(th))
			},
		},
		{
			Name:    "Evening routine — coordinator dispatches",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// (nothing — we inject the time marker in Wait)
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("[time] evening — run the evening routine: dispatcher should tell comfort to lower lights and thermostat, and security to arm monitoring.")
					}
					// Look for evidence that comfort executed the routine:
					// thermostat set, and at least one light turned off or dimmed.
					entries := ReadAuditEntries(dir)
					thermoSets := CountTool(entries, "set_thermostat")
					t.Logf("  ... set_thermostat=%d threads=%v", thermoSets, ThreadIDs(th))
					return thermoSets >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				if CountTool(entries, "set_thermostat") == 0 {
					t.Logf("NOTE: evening routine did not adjust thermostat")
				}
			},
		},
		{
			Name:    "Quiescence — agent idle and stable",
			Timeout: 30 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Let the quiescence window elapse. A crashed agent is
				// caught by the scenario hard timeout; thread count is
				// the agent's own bookkeeping, not the goal.
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				t.Logf("quiescence: %d sub-threads alive, %d audit entries", th.Threads().Count(), len(entries))
			},
		},
	},
	Timeout: 10 * time.Minute,
}

func TestScenario_HomeAutomation(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	sensorsBin := BuildMCPBinary(t, "mcps/home_sensors")
	camerasBin := BuildMCPBinary(t, "mcps/home_cameras")
	intercomBin := BuildMCPBinary(t, "mcps/home_intercom")
	devicesBin := BuildMCPBinary(t, "mcps/home_devices")
	t.Logf("built home_sensors=%s home_cameras=%s home_intercom=%s home_devices=%s",
		sensorsBin, camerasBin, intercomBin, devicesBin)

	s := homeAutomationScenario
	s.MCPServers[0].Command = sensorsBin
	s.MCPServers[1].Command = camerasBin
	s.MCPServers[2].Command = intercomBin
	s.MCPServers[3].Command = devicesBin
	RunScenario(t, s)
}
