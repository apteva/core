// Run only simulated-domain core evaluations, using the current global Codex
// connection from an explicitly selected local Apteva data directory.
// Credentials are decrypted in memory and passed through the child environment.
import { Database } from "bun:sqlite";
import { createDecipheriv } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const dataDir = process.env.APTEVA_DATA_DIR;
if (!dataDir) throw new Error("Set APTEVA_DATA_DIR to the local Apteva directory containing apteva.db and .secret");
const db = new Database(resolve(dataDir, "apteva.db"), { readonly: true });
type Connection = { id: number; encrypted_credentials: string; runtime_config: string };
const connection = db.query(`SELECT id, encrypted_credentials, runtime_config
  FROM connections WHERE app_slug = 'openai-codex' AND project_id = ''
  ORDER BY is_primary DESC, id LIMIT 1`).get() as Connection | null;
db.close();
if (!connection) throw new Error("No global OpenAI Codex connection in the selected Apteva database");
const key = Buffer.from(readFileSync(resolve(dataDir, ".secret"), "utf8").trim(), "hex");
const encrypted = Buffer.from(connection.encrypted_credentials, "hex");
const decipher = createDecipheriv("aes-256-gcm", key, encrypted.subarray(0, 12));
decipher.setAuthTag(encrypted.subarray(-16));
const credentials = JSON.parse(Buffer.concat([decipher.update(encrypted.subarray(12, -16)), decipher.final()]).toString());
const runtime = JSON.parse(connection.runtime_config);
const model = process.env.PROACTIVITY_MODEL?.trim() || runtime.model_large;
const reasoning = process.env.PROACTIVITY_REASONING?.trim() || "";
if (reasoning && !["auto", "none", "minimal", "low", "medium", "high", "xhigh"].includes(reasoning)) throw new Error("Invalid PROACTIVITY_REASONING");
if (!credentials.access_token || !model) throw new Error("Codex access token or selected model is missing");
if (credentials.expires_at && Date.parse(credentials.expires_at) <= Date.now()) throw new Error("Codex token expired; reauthenticate in the selected Apteva server first");
const reports = resolve(process.env.PROACTIVITY_REPORT_DIR ?? `test-artifacts/proactivity-${new Date().toISOString().replaceAll(":", "-")}`);
mkdirSync(reports, { recursive: true, mode: 0o700 });
const filter = process.argv[2] ?? "^TestCodexProactivity";
console.log(JSON.stringify({ connection: connection.id, model, reasoning: reasoning || "adaptive (medium baseline)", filter, reports }));
const child = Bun.spawn([
  process.env.GO_BINARY ?? "go", "test", ".", "-run", filter,
  "-count=1", "-timeout=60m", "-v",
], {
  cwd: resolve(import.meta.dir, ".."),
  env: {
    ...process.env,
    GOWORK: "off",
    RUN_CODEX_PROACTIVITY: "1",
    OPENAI_CODEX_ACCESS_TOKEN: credentials.access_token,
    OPENAI_CODEX_ACCOUNT_ID: credentials.account_id ?? "",
    PROACTIVITY_MODEL: model,
    PROACTIVITY_REASONING: reasoning,
    PROACTIVITY_REPORT_DIR: reports,
  },
  stdout: "inherit", stderr: "inherit",
});
// Avoid leaving inference processes running if the operator interrupts the runner.
for (const signal of ["SIGINT", "SIGTERM"] as const) process.on(signal, () => child.kill(signal));
process.exit(await child.exited);
