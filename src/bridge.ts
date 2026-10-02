/**
 * lavra-pi bridge extension
 *
 * One `pi install git:github.com/roberto-mello/lavra-pi` gets you:
 * - 30 specialized agents (from @lavralabs/lavra npm dependency)
 * - 15+ skills (loaded via pi's native Agent Skills support)
 * - Auto-recall and memory-capture (replacing Claude Code hooks)
 * - FTS5 knowledge search via SQLite (BM25-ranked, matching knowledge-db.sh)
 * - Context7 framework docs (direct HTTP, no MCP server)
 * - Web search (Brave API or agent-browser)
 * - Custom `lavra_subagent` tool with single/parallel/chain modes
 * - 28 /lavra-* commands
 *
 * Dependencies:
 *   - @lavralabs/lavra — provides agents/, skills/, hooks/ (npm package)
 *   - sqlite3 CLI — for FTS5 knowledge search (already required by Lavra)
 *   - Optional: BRAVE_API_KEY env var for web search
 */

import { spawn, execFileSync, execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { StringEnum } from "@earendil-works/pi-ai";
import {
  type ExtensionAPI,
  type ExtensionContext,
  isToolCallEventType,
} from "@earendil-works/pi-coding-agent";
import { Key, matchesKey, truncateToWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import { Type } from "typebox";

// ═══════════════════════════════════════════════════
// Path resolution from @lavralabs/lavra npm package
// ═══════════════════════════════════════════════════

/** Resolve the lavra package root inside node_modules */
function lavraPackageRoot(): string {
  // Prefer a checked-out Lavra source tree for local development. Set
  // LAVRA_SOURCE_DIR to override the conventional ~/Documents/projects/lavra
  // location; installed npm packages remain the fallback for other machines.
  const localSource = process.env.LAVRA_SOURCE_DIR?.trim();
  const candidates = [
    ...(localSource ? [path.resolve(localSource)] : []),
    path.join(os.homedir(), "Documents/projects/lavra"),
    // When pi installs a git package and runs npm install, @lavralabs/lavra
    // ends up in node_modules/@lavralabs/lavra/ relative to this package root.
    path.resolve(__dirname, "../node_modules/@lavralabs/lavra"),
    path.resolve(__dirname, "../../@lavralabs/lavra"),
    // When running via pi -e for local dev, the package might be resolved differently
    path.resolve(__dirname, "../../node_modules/@lavralabs/lavra"),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, "plugins/lavra/agents"))) return dir;
  }
  throw new Error(
    "Cannot find @lavralabs/lavra package. Is it installed?\n" +
      "Run: pi install git:github.com/roberto-mello/lavra-pi",
  );
}

function lavraPluginsPath(): string {
  return path.join(lavraPackageRoot(), "plugins/lavra");
}

const AGENTS_DIR = path.join(lavraPluginsPath(), "agents");
const HOOKS_DIR = path.join(lavraPluginsPath(), "hooks");
const MEMORY_SUBDIR = ".lavra/memory";

// ═══════════════════════════════════════════════════
// Project root resolution (matches Lavra's bash scripts)
// ═══════════════════════════════════════════════════

/**
 * Walk up from cwd to find the project root (where .beads/ or .lavra/ lives).
 * Falls back to cwd if not found.
 *
 * Matches the logic in recall.sh and auto-recall.sh.
 */
function findProjectRoot(cwd: string): string {
  let root = cwd;
  while (root !== "/") {
    if (fs.existsSync(path.join(root, ".beads")) ||
        fs.existsSync(path.join(root, ".lavra"))) {
      return root;
    }
    root = path.dirname(root);
  }
  return cwd; // fallback: no .lavra found
}

/** Check whether cwd lives in a Lavra-enabled project */
function isLavraProject(cwd: string): boolean {
  const root = findProjectRoot(cwd);
  return fs.existsSync(path.join(root, ".beads")) ||
         fs.existsSync(path.join(root, ".lavra"));
}

// ═══════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════

interface AgentDef {
  name: string;
  description: string;
  model: string;
  tools: string[];
  color: string;
  category: string;
  systemPrompt: string;
}

interface KnowledgeEntry {
  key: string;
  type: string;
  content: string;
  source: string;
  tags: string[];
  ts: number;
  bead: string;
}

interface LavraModelConfig {
  fast?: string;
  default?: string;
  quality?: string;
  // Claude-era aliases retained for existing agent definitions/configs.
  haiku?: string;
  sonnet?: string;
  opus?: string;
}

const LAVRA_CONFIG_RELATIVE = ".lavra/config/lavra.json";
const MAX_AGENT_OUTPUT_CHARS = 24_000;

function lavraConfigPath(projectRoot: string): string {
  return path.join(projectRoot, LAVRA_CONFIG_RELATIVE);
}

function readLavraConfig(projectRoot: string): Record<string, any> {
  try {
    const config = JSON.parse(fs.readFileSync(lavraConfigPath(projectRoot), "utf-8"));
    return config && typeof config === "object" && !Array.isArray(config) ? config : {};
  } catch {
    return {};
  }
}

function readModelConfig(projectRoot: string): LavraModelConfig {
  const models = readLavraConfig(projectRoot).models;
  return models && typeof models === "object" && !Array.isArray(models) ? models : {};
}

function writeModelConfig(projectRoot: string, models: LavraModelConfig): void {
  const file = lavraConfigPath(projectRoot);
  const config = readLavraConfig(projectRoot);
  config.models = models;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(config, null, 2) + "\n", "utf-8");
}

function maxParallelAgents(projectRoot: string): number {
  const value = readLavraConfig(projectRoot)?.execution?.max_parallel_agents;
  return Number.isInteger(value) && value > 0 ? value : 3;
}

function boundedAgentOutput(output: string): string {
  return output.length <= MAX_AGENT_OUTPUT_CHARS
    ? output
    : output.slice(0, MAX_AGENT_OUTPUT_CHARS) + "\n\n[Subagent output truncated by lavra-pi.]";
}

// ═══════════════════════════════════════════════════
// Agent Discovery (from @lavralabs/lavra npm package)
// ═══════════════════════════════════════════════════

function discoverAgents(): AgentDef[] {
  const agents: AgentDef[] = [];
  const categories = ["review", "research", "design", "workflow", "docs"];
  for (const cat of categories) {
    const dir = path.join(AGENTS_DIR, cat);
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir)) {
      if (!file.endsWith(".md")) continue;
      const content = fs.readFileSync(path.join(dir, file), "utf-8");
      const fm = parseFrontmatter(content);
      if (!fm?.name) continue;
      agents.push({
        name: fm.name,
        description: fm.description ?? "",
        model: fm.model ?? "inherit",
        tools: fm.tools?.split(",").map((s: string) => s.trim()).filter(Boolean) ?? [],
        color: fm.color ?? "blue",
        category: cat,
        systemPrompt: fm._body ?? content,
      });
    }
  }
  return agents;
}

function parseFrontmatter(content: string): Record<string, any> | null {
  const m = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!m) return null;
  const fm: Record<string, any> = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^(\w+):\s*(.*)/);
    if (kv) fm[kv[1]] = kv[2].trim();
  }
  fm._body = content.slice(m[0].length);
  return fm;
}

// ═══════════════════════════════════════════════════
// Knowledge Store — Full FTS5 via sqlite3 CLI
//
// Replicates the logic from knowledge-db.sh:
//   - SQLite FTS5 with BM25 ranking
//   - Incremental JSONL → SQLite sync
//   - First-time backfill from beads comments
//   - Deduplication by key
// ═══════════════════════════════════════════════════

function memoryDir(projectRoot: string): string {
  return path.join(projectRoot, MEMORY_SUBDIR);
}

function knowledgeDbPath(projectRoot: string): string {
  return path.join(memoryDir(projectRoot), "knowledge.db");
}

function knowledgeJsonlPath(projectRoot: string): string {
  return path.join(memoryDir(projectRoot), "knowledge.jsonl");
}

function ensureMemoryDir(projectRoot: string): void {
  fs.mkdirSync(memoryDir(projectRoot), { recursive: true });
}

/** Check if sqlite3 CLI is available */
function hasSqlite3(): boolean {
  try {
    execSync("sqlite3 --version", { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/** Run a sqlite3 command. Args after the SQL are passed as positional params. */
function sqlite<T = string>(
  dbPath: string,
  sql: string,
  ...args: string[]
): string {
  const cliArgs = [dbPath, ...args.flatMap((a) => ["-cmd", a]), sql];
  return String(execSync("sqlite3", cliArgs, { encoding: "utf-8", timeout: 10000 })).trim();
}

/** Ensure the SQLite FTS5 schema exists */
function ensureDbSchema(dbPath: string): void {
  const dir = path.dirname(dbPath);
  fs.mkdirSync(dir, { recursive: true });

  sqlite(dbPath, `
    CREATE TABLE IF NOT EXISTS knowledge(
      key TEXT PRIMARY KEY,
      type TEXT,
      content TEXT,
      source TEXT,
      tags_text TEXT,
      ts INTEGER,
      bead TEXT
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
      content, tags_text, type, key,
      content=knowledge,
      content_rowid=rowid,
      tokenize='porter unicode61'
    );
    CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
      INSERT INTO knowledge_fts(rowid, content, tags_text, type, key)
      VALUES (new.rowid, new.content, new.tags_text, new.type, new.key);
    END;
  `);
}

/** Insert one entry via CSV (zero SQL injection risk, matches knowledge-db.sh) */
function insertEntry(dbPath: string, entry: KnowledgeEntry): void {
  // Deduplicate by key
  const exists = sqlite(
    dbPath,
    `SELECT count(*) FROM knowledge WHERE key = ?`,
    `-cmd`, `.parameter set $1 ${entry.key}`,
  );
  if (exists !== "0") return;

  const tmpFile = path.join(os.tmpdir(), `kb-${Date.now()}-${Math.random().toString(36).slice(2)}.csv`);
  try {
    const tagsText = entry.tags.join(" ");
    // jq-based CSV escaping (same as knowledge-db.sh)
    const csvLine = execSync(
      `jq -nr --arg key ${JSON.stringify(entry.key)} --arg type ${JSON.stringify(entry.type)} ` +
        `--arg content ${JSON.stringify(entry.content)} --arg source ${JSON.stringify(entry.source)} ` +
        `--arg tags_text ${JSON.stringify(tagsText)} --argjson ts ${entry.ts} ` +
        `--arg bead ${JSON.stringify(entry.bead)} '[$key, $type, $content, $source, $tags_text, $ts, $bead] | @csv'`,
      { encoding: "utf-8", timeout: 5000 },
    ).toString().trim();
    fs.writeFileSync(tmpFile, csvLine + "\n", "utf-8");
    execSync(`sqlite3 "${dbPath}" ".mode csv" ".import '${tmpFile}' knowledge"`, {
      stdio: "ignore",
      timeout: 10000,
    });
  } finally {
    try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
  }
}

/** Sync JSONL file entries into SQLite (incremental, deduplicated) */
function syncJsonlToDb(dbPath: string, jsonlPath: string): void {
  if (!fs.existsSync(jsonlPath)) return;
  const content = fs.readFileSync(jsonlPath, "utf-8").trim();
  if (!content) return;

  for (const line of content.split("\n").filter(Boolean)) {
    try {
      const entry = JSON.parse(line) as KnowledgeEntry;
      if (!entry.key) continue;
      insertEntry(dbPath, entry);
    } catch { /* skip malformed lines */ }
  }
}

/** Full sync: ensure schema, backfill from beads (first time), sync JSONL files */
function syncKnowledge(projectRoot: string): void {
  const dbPath = knowledgeDbPath(projectRoot);
  ensureMemoryDir(projectRoot);
  ensureDbSchema(dbPath);

  const count = sqlite(dbPath, "SELECT count(*) FROM knowledge;");
  if (count === "0") {
    // First-time backfill from beads comments (matches kb_sync logic)
    try {
      const comments = execSync(
        `bd sql --json "SELECT issue_id, text FROM comments WHERE text LIKE 'LEARNED:%' OR text LIKE 'DECISION:%' OR text LIKE 'FACT:%' OR text LIKE 'PATTERN:%' OR text LIKE 'INVESTIGATION:%'"`,
        { encoding: "utf-8", timeout: 15000 },
      ).toString().trim();
      if (comments && comments !== "[]") {
        const rows = JSON.parse(comments);
        for (const row of rows) {
          for (const prefix of ["INVESTIGATION", "LEARNED", "DECISION", "FACT", "PATTERN"]) {
            const text: string = row.text ?? "";
            if (!text.startsWith(prefix + ":")) continue;
            const type = prefix.toLowerCase();
            const content = text.slice(prefix.length + 1).trim().slice(0, 2048);
            const slug = content.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
            insertEntry(dbPath, {
              key: `${type}-${slug}`,
              type,
              content,
              source: "backfill",
              tags: [type],
              ts: Math.floor(Date.now() / 1000),
              bead: row.issue_id ?? "",
            });
          }
        }
      }
    } catch { /* bd not available or no comments — proceed */ }
  }

  // Sync JSONL files
  syncJsonlToDb(dbPath, knowledgeJsonlPath(projectRoot));
  const archivePath = path.join(memoryDir(projectRoot), "knowledge.archive.jsonl");
  syncJsonlToDb(dbPath, archivePath);
}

/** FTS5 search with BM25 ranking (matches kb_search from knowledge-db.sh) */
function searchKnowledge(
  projectRoot: string,
  query: string,
  limit = 10,
): Array<{ type: string; content: string; bead: string; tags: string }> {
  const dbPath = knowledgeDbPath(projectRoot);
  if (!fs.existsSync(dbPath)) return [];

  // Extract 2+ char alphanumeric terms (same sanitization as knowledge-db.sh)
  const terms = query.match(/\b[a-zA-Z0-9_.]{2,}\b/g);
  if (!terms || terms.length === 0) return [];

  // Build FTS5 MATCH expression: quoted terms joined by OR
  const ftsQuery = terms.map((t) => `"${t.replace(/"/g, '""')}"`).join(" OR ");

  // BM25 weights: content=-10, tags_text=-5, type=-2, key=-1
  try {
    const result = sqlite(
      dbPath,
      `.separator "|"
       SELECT k.type, k.content, k.bead, k.tags_text
       FROM knowledge_fts fts
       JOIN knowledge k ON k.rowid = fts.rowid
       WHERE knowledge_fts MATCH '${ftsQuery.replace(/'/g, "''")}'
       ORDER BY bm25(knowledge_fts, -10.0, -5.0, -2.0, -1.0)
       LIMIT ${Math.min(Math.max(1, limit), 50)};`,
    );
    if (!result) return [];

    return result.split("\n").filter(Boolean).map((line) => {
      const [type, content, bead, tags] = line.split("|");
      return { type: type ?? "", content: content ?? "", bead: bead ?? "", tags: tags ?? "" };
    });
  } catch {
    return [];
  }
}

/** Append to JSONL (for new entries from memory capture). Rotates at 5000 lines. */
function appendKnowledgeJsonl(projectRoot: string, entry: KnowledgeEntry): void {
  const file = knowledgeJsonlPath(projectRoot);
  ensureMemoryDir(projectRoot);

  // Deduplicate by key
  if (fs.existsSync(file)) {
    const existing = fs.readFileSync(file, "utf-8");
    if (existing.includes(`"key":"${entry.key}"`)) return;
    fs.appendFileSync(file, JSON.stringify(entry) + "\n", "utf-8");

    // Rotation at 5000 lines (matches memory-capture.sh)
    const lines = existing.split("\n").filter(Boolean).length + 1;
    if (lines > 5000) {
      const allLines = fs.readFileSync(file, "utf-8").trim().split("\n");
      const archive = path.join(memoryDir(projectRoot), "knowledge.archive.jsonl");
      fs.appendFileSync(archive, allLines.slice(0, 2500).join("\n") + "\n");
      fs.writeFileSync(file, allLines.slice(2500).join("\n") + "\n");
    }
  } else {
    fs.writeFileSync(file, JSON.stringify(entry) + "\n", "utf-8");
  }

  // Also write to SQLite
  try {
    const dbPath = knowledgeDbPath(projectRoot);
    if (fs.existsSync(dbPath)) {
      insertEntry(dbPath, entry);
    }
  } catch { /* sqlite not available — JSONL-only is fine */ }
}

/** Read session state file (one-shot after compaction recovery) */
function readSessionState(projectRoot: string): string | null {
  const stateFile = path.join(memoryDir(projectRoot), "session-state.md");
  if (!fs.existsSync(stateFile)) return null;
  const stat = fs.statSync(stateFile);
  const ageSec = (Date.now() - stat.mtimeMs) / 1000;
  if (ageSec > 86400) {
    fs.unlinkSync(stateFile);
    return null;
  }
  const content = fs.readFileSync(stateFile, "utf-8").slice(0, 10000);
  fs.unlinkSync(stateFile);
  return content;
}

// ═══════════════════════════════════════════════════
// Hook: capture knowledge from bd comments add
// ═══════════════════════════════════════════════════

function captureKnowledgeFromBashCommand(
  projectRoot: string,
  command: string,
): void {
  const pattern =
    /bd\s+comments?\s+add\s+([A-Za-z0-9._-]+)\s+["'](INVESTIGATION|LEARNED|DECISION|FACT|PATTERN|DEVIATION|MUST-CHECK):\s*(.*?)["']/i;
  const m = command.match(pattern);
  if (!m) return;
  if (m[2].toUpperCase() === "SKIP") return;

  const beadId = m[1];
  const type = m[2].toLowerCase().replace("_", "-");
  const content = m[3].slice(0, 2048);
  const slug = content.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
  const key = `${type}-${slug}`;

  appendKnowledgeJsonl(projectRoot, {
    key,
    type,
    content,
    source: "user",
    tags: [type],
    ts: Math.floor(Date.now() / 1000),
    bead: beadId,
  });
}

// ═══════════════════════════════════════════════════
// Subagent Execution (spawns pi --mode json subprocess)
// ═══════════════════════════════════════════════════

type SubagentTranscriptKind = "assistant" | "thinking" | "tool" | "status";

interface SubagentTranscriptEvent {
  kind: SubagentTranscriptKind;
  text: string;
  append?: boolean;
}

interface SubagentSession {
  label: string;
  task: string;
  status: string;
  entries: Array<{ kind: SubagentTranscriptKind; text: string }>;
}

class SubagentSessionStore {
  private readonly sessions: SubagentSession[] = [];
  private readonly listeners = new Set<() => void>();

  create(label: string, task: string): SubagentSession {
    const session = { label, task, status: "starting", entries: [] };
    this.sessions.push(session);
    this.notify();
    return session;
  }

  getSessions(): readonly SubagentSession[] { return this.sessions; }

  updateStatus(session: SubagentSession, text: string): void {
    const status = text.startsWith("responding:") ? "responding" : text;
    if (session.status === status) return;
    session.status = status;
    this.notify();
  }

  append(session: SubagentSession, event: SubagentTranscriptEvent): void {
    if (!event.text) return;
    const last = session.entries[session.entries.length - 1];
    if (event.append && last?.kind === event.kind) last.text += event.text;
    else session.entries.push({ kind: event.kind, text: event.text });
    if (session.entries.length > 1000) session.entries.splice(0, session.entries.length - 1000);
    this.notify();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void { for (const listener of this.listeners) listener(); }
}

function availableSkillDirectories(cwd: string): string[] {
  const projectRoot = findProjectRoot(cwd);
  const roots = [cwd, projectRoot, os.homedir()];
  const suffixes = [".pi/skills", ".agents/skills", ".claude/skills", ".codex/skills"];
  const localSourceSkills = [
    path.join(lavraPluginsPath(), "skills"),
    path.join(lavraPackageRoot(), "plugins/lavra/codex/skills"),
  ];
  return [...new Set([
    ...localSourceSkills,
    ...roots.flatMap((root) => suffixes.map((suffix) => path.join(root, suffix))),
  ])].filter((dir) => fs.existsSync(dir));
}

async function runAgent(
  agent: AgentDef,
  task: string,
  cwd: string,
  signal?: AbortSignal,
  onProgress?: (text: string) => void,
  onTranscript?: (event: SubagentTranscriptEvent) => void,
  modelOverride?: string,
): Promise<{ output: string; error?: string; usage: any }> {
  // Children must be real one-shot workers: no inherited project instructions,
  // no high global thinking default, and no second "continue" turn.
  const args: string[] = [
    "--mode", "json", "-p", "--no-session",
    "--no-context-files",
    "--exclude-tools", "lavra_subagent",
    "--thinking", "medium",
  ];
  for (const skillsDir of availableSkillDirectories(cwd)) args.push("--skill", skillsDir);
  if (modelOverride) args.push("--model", modelOverride);
  const defaultTools = agent.category === "review"
    ? ["read", "bash", "grep", "find", "ls"]
    : agent.category === "research"
      ? ["read", "bash", "grep", "find", "ls", "lavra_web_search", "web_search", "framework_docs", "knowledge_search"]
      : ["read", "bash", "edit", "write", "grep", "find", "ls"];
  args.push("--tools", (agent.tools?.length ? agent.tools : defaultTools).join(","));

  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "lavra-agent-"));
  const promptFile = path.join(tmpDir, "prompt.md");
  try {
    const fullPrompt = `${agent.systemPrompt}\n\nTask: ${task}`;
    await fs.promises.writeFile(promptFile, fullPrompt, { encoding: "utf-8", mode: 0o600 });
    // Pi reads files only through the @file syntax. Passing the path bare
    // makes the child prompt be the filename rather than fullPrompt.
    args.push(`@${promptFile}`);
  } catch {
    // fallback: pass task inline
    args.push(`Task: ${task}`);
  }

  return new Promise((resolve) => {
    const proc = spawn("pi", args, {
      cwd,
      shell: false,
      // Mark children so the bridge can disable recursive lavra_subagent
      // registration while retaining the useful search/documentation tools.
      env: { ...process.env, LAVRA_PI_SUBAGENT: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let liveText = "";
    const messages: any[] = [];
    let buffer = "";

    proc.stdout.on("data", (data: Buffer) => {
      buffer += data.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const evt = JSON.parse(line);
          if (evt.type === "message_update") {
            const update = evt.assistantMessageEvent;
            if (update?.type === "text_delta" && update.delta) {
              liveText += update.delta;
              onTranscript?.({ kind: "assistant", text: update.delta, append: true });
              onProgress?.(`responding: ${liveText.slice(-180)}`);
            } else if (update?.type === "thinking_start") {
              onTranscript?.({ kind: "thinking", text: "Thinking..." });
              onProgress?.("thinking...");
            } else if (update?.type === "thinking_delta" && update.delta) {
              onTranscript?.({ kind: "thinking", text: update.delta, append: true });
              onProgress?.("thinking...");
            } else if (update?.type === "toolcall_start") {
              onTranscript?.({ kind: "tool", text: "Preparing tool call..." });
              onProgress?.("preparing tool call...");
            }
          } else if (evt.type === "tool_execution_start") {
            onTranscript?.({ kind: "tool", text: `Running ${evt.toolName}` });
            onProgress?.(`running ${evt.toolName}`);
          } else if (evt.type === "tool_execution_end") {
            onTranscript?.({ kind: "tool", text: `${evt.isError ? "Failed" : "Finished"} ${evt.toolName}` });
            onProgress?.(`${evt.isError ? "failed" : "finished"} ${evt.toolName}`);
          } else if (evt.type === "message_end" && evt.message) {
            messages.push(evt.message);
            for (const part of evt.message.content ?? []) {
              if (part.type === "text") {
                stdout = part.text;
                liveText = part.text;
                if (evt.message.role === "toolResult") onTranscript?.({ kind: "tool", text: part.text });
              }
            }
          }
        } catch { /* non-JSON progress output — ignore */ }
      }
    });
    proc.stderr.on("data", (data: Buffer) => { stderr += data.toString(); });

    proc.on("close", (code) => {
      // Aggressive cleanup
      try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch { /* ignore */ }

      const usage = {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: messages.length,
      };
      for (const msg of messages) {
        if (msg.usage) {
          usage.input += msg.usage.input || 0;
          usage.output += msg.usage.output || 0;
          usage.cacheRead += msg.usage.cacheRead || 0;
          usage.cacheWrite += msg.usage.cacheWrite || 0;
          usage.cost += msg.usage.cost?.total || 0;
        }
      }
      resolve(code === 0
        ? { output: stdout, usage }
        : { output: stdout, error: stderr || `exit ${code}`, usage });
    });
    proc.on("error", (err) => {
      try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch { /* ignore */ }
      resolve({ output: "", error: err.message, usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 } });
    });

    if (signal) {
      const kill = () => {
        proc.kill("SIGTERM");
        setTimeout(() => { if (!proc.killed) proc.kill("SIGKILL"); }, 5000);
      };
      if (signal.aborted) kill();
      else signal.addEventListener("abort", kill, { once: true });
    }
  });
}

interface SubagentProgressSink {
  start(label: string, task: string): SubagentSession;
  update(session: SubagentSession, text: string): void;
  transcript(session: SubagentSession, event: SubagentTranscriptEvent): void;
}

class SubagentSessionViewer {
  private selected = 0;
  private scrollFromBottom = 0;

  constructor(
    private readonly store: SubagentSessionStore,
    private readonly theme: any,
    private readonly onClose: () => void,
    private readonly requestRender: () => void,
  ) {}

  render(width: number): string[] {
    const sessions = this.store.getSessions();
    if (sessions.length === 0) return [this.theme.fg("muted", "No subagent sessions are active."), "", "↑ return to the main session"];
    const session = sessions[Math.min(this.selected, sessions.length - 1)];
    const header = this.theme.fg("accent", `Lavra subagent ${this.selected + 1}/${sessions.length} — ${session.label} — ${session.status}`);
    const help = this.theme.fg("dim", "←/→ switch subagent • ↑ return to main • PgUp/PgDn scroll");
    const task = session.task ? this.theme.fg("muted", `Task: ${session.task}`) : "";
    const contentWidth = Math.max(20, width - 2);
    const rows: string[] = [header, help, task, ""];
    for (const entry of session.entries) {
      const color = entry.kind === "assistant" ? "text" : entry.kind === "thinking" ? "thinkingHigh" : entry.kind === "tool" ? "warning" : "dim";
      const prefix = entry.kind === "assistant" ? "" : `[${entry.kind}] `;
      rows.push(...wrapTextWithAnsi(this.theme.fg(color, prefix + entry.text), contentWidth));
    }
    const maxRows = 30;
    const end = Math.max(maxRows, rows.length - this.scrollFromBottom);
    return rows.slice(Math.max(0, end - maxRows), end).map((line) => truncateToWidth(line, width));
  }

  handleInput(data: string): void {
    const sessions = this.store.getSessions();
    if (matchesKey(data, Key.up) || matchesKey(data, Key.escape)) {
      this.onClose();
      return;
    }
    if (matchesKey(data, Key.right) && sessions.length > 1) {
      this.selected = (this.selected + 1) % sessions.length;
      this.scrollFromBottom = 0;
    } else if (matchesKey(data, Key.left) && sessions.length > 1) {
      this.selected = (this.selected + sessions.length - 1) % sessions.length;
      this.scrollFromBottom = 0;
    } else if (matchesKey(data, Key.pageUp) || matchesKey(data, Key.home)) {
      this.scrollFromBottom = Math.min(this.currentEntryCount(), this.scrollFromBottom + 20);
    } else if (matchesKey(data, Key.pageDown) || matchesKey(data, Key.end) || matchesKey(data, Key.down)) {
      this.scrollFromBottom = Math.max(0, this.scrollFromBottom - (matchesKey(data, Key.down) ? 1 : 20));
    } else {
      return;
    }
    this.requestRender();
  }

  invalidate(): void {}

  private currentEntryCount(): number {
    return this.store.getSessions()[this.selected]?.entries.length ?? 0;
  }
}

// ═══════════════════════════════════════════════════
// External API integrations
// ═══════════════════════════════════════════════════

async function fetchContext7Docs(query: string, limit = 5): Promise<string> {
  try {
    const res = await fetch("https://mcp.context7.com/mcp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0", id: 1,
        method: "search_documentation",
        params: { query, limit },
      }),
    });
    const data: any = await res.json();
    if (data?.result?.results?.length) {
      return data.result.results
        .map((r: any) => `### [${r.title}](${r.url})\n\n${r.content?.slice(0, 1000)}`)
        .join("\n\n");
    }
    return "No documentation results from Context7.";
  } catch (err: any) {
    return `Context7 lookup failed: ${err.message}`;
  }
}

async function webSearch(query: string): Promise<string> {
  const apiKey = process.env.BRAVE_API_KEY;
  if (apiKey) {
    try {
      const res = await fetch(
        `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=5`,
        { headers: { Accept: "application/json", "X-Subscription-Token": apiKey } },
      );
      if (!res.ok) return `Search API returned ${res.status}`;
      const data: any = await res.json();
      return (data.web?.results ?? [])
        .map((r: any) => `- [${r.title}](${r.url}): ${r.description}`)
        .join("\n");
    } catch (err: any) {
      return `Search error: ${err.message}`;
    }
  }
  return "No BRAVE_API_KEY set. Set it in your environment or use agent-browser for web search.";
}

// ═══════════════════════════════════════════════════
// EXTENSION ENTRY POINT
// ═══════════════════════════════════════════════════

export default function (pi: ExtensionAPI) {
  // Show lavra-pi loaded version in startup header
  let lavraVersion = "unknown";
  try {
    const pkg = JSON.parse(
      fs.readFileSync(path.join(lavraPackageRoot(), "package.json"), "utf-8"),
    );
    lavraVersion = pkg.version ?? lavraVersion;
  } catch { /* ignore */ }

  // ── Load agents from @lavralabs/lavra npm package ──
  let agents: AgentDef[] = [];
  try {
    agents = discoverAgents();
  } catch (err: any) {
    // Will surface at first agent tool use if @lavralabs/lavra isn't installed
    agents = [];
  }

  function activeModelId(ctx: ExtensionContext): string | undefined {
    return ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;
  }

  function resolveAgentModel(
    agent: AgentDef,
    ctx: ExtensionContext,
    override?: string,
  ): string | undefined {
    const requested = (override || agent.model || "inherit").trim().toLowerCase();
    const current = activeModelId(ctx);
    if (!requested || requested === "inherit" || requested === "current") return current;

    const tier = requested === "haiku" ? "fast"
      : requested === "sonnet" || requested === "opus" ? "quality"
        : requested === "fast" || requested === "default" || requested === "quality" ? requested
          : undefined;
    if (tier) {
      const models = readModelConfig(findProjectRoot(ctx.cwd));
      const configured = models[tier] || models[requested as keyof LavraModelConfig];
      return configured && configured !== "inherit" && configured !== "current"
        ? configured
        : current;
    }

    // Explicit Pi model IDs are passed through unchanged.
    return requested.includes("/") ? override || agent.model : current;
  }

  // Make project/user skills available to the main session and every child Pi.
  pi.on("resources_discover", (event) => ({
    skillPaths: availableSkillDirectories(event.cwd),
  }));

  // ════════════════════════════════════════════
  //  1. SESSION_START → Auto-Recall
  // ════════════════════════════════════════════

  pi.on("session_start", async (_event, ctx) => {
    // Quick check: is this a Lavra project?
    if (!isLavraProject(ctx.cwd)) return;
    const projectRoot = findProjectRoot(ctx.cwd);

    try {
      // Sync knowledge JSONL → SQLite FTS5
      if (hasSqlite3()) {
        syncKnowledge(projectRoot);
      }

      // Recover session state
      const sessionState = readSessionState(projectRoot);

      // Recall relevant knowledge
      const knowledgeFile = knowledgeJsonlPath(projectRoot);
      let knowledgeContext = "";
      if (fs.existsSync(knowledgeFile)) {
        // Read recent entries for general context
        const recent = fs.readFileSync(knowledgeFile, "utf-8")
          .trim().split("\n").filter(Boolean).slice(-10);
        if (recent.length > 0) {
          const entries = recent.map((l) => {
            try {
              const e = JSON.parse(l) as KnowledgeEntry;
              return `${e.type.toUpperCase()}: ${e.content}`;
            } catch { return null; }
          }).filter(Boolean).join("\n");
          if (entries) {
            knowledgeContext =
              "## Relevant Knowledge from Memory\n\n" + entries +
              "\n\n_Use `/lavra-recall <query>` for FTS5 search._\n";
          }
        }
      }

      // Build context message
      const parts: string[] = [];
      if (sessionState) {
        parts.push(
          "## Session State (recovered after compaction)\n\n" +
          sessionState + "\n",
        );
      }
      if (knowledgeContext) parts.push(knowledgeContext);
      if (parts.length === 0) {
        parts.push(
          "## Lavra is ready.\n\n" +
          "| Goal | Command |\n" +
          "|------|---------|\n" +
          "| New feature | `/lavra-brainstorm <desc>` |\n" +
          "| Plan from spec | `/lavra-design <desc>` |\n" +
          "| Existing beads | `/lavra-work` |\n" +
          "| Explore ideas | `/lavra-brainstorm <idea>` |\n" +
          "| Search knowledge | `/lavra-recall <query>` |\n",
        );
      }

      ctx.ui.notify(`Lavra ${lavraVersion}: context loaded (${agents.length} agents)`, "info");
    } catch (err: any) {
      ctx.ui.notify(`Lavra init error: ${err.message}`, "error");
    }
  });

  // ════════════════════════════════════════════
  //  2. TOOL_RESULT → Memory Capture
  // ════════════════════════════════════════════

  pi.on("tool_result", (event, ctx) => {
    if (!isToolCallEventType("bash", event)) return;
    if (!event.input?.command) return;
    if (!isLavraProject(ctx.cwd)) return;
    const projectRoot = findProjectRoot(ctx.cwd);
    captureKnowledgeFromBashCommand(projectRoot, event.input.command);
  });

  // ════════════════════════════════════════════
  //  3. /lavra-* COMMANDS
  // ════════════════════════════════════════════

  function hasLavraProject(cwd: string): boolean {
    return isLavraProject(cwd);
  }

  function skillFilePath(name: string, cwd?: string): string | null {
    // Skill names come from controlled skill metadata; reject path traversal.
    if (!/^[a-z0-9][a-z0-9-]*$/.test(name)) return null;

    const projectRoots = cwd ? [cwd, findProjectRoot(cwd)] : [];
    const dirs = [".pi/skills", ".agents/skills", ".claude/skills", ".codex/skills"];
    for (const root of [...new Set(projectRoots)]) {
      for (const dir of dirs) {
        const file = path.join(root, dir, name, "SKILL.md");
        if (fs.existsSync(file)) return file;
      }
    }

    // Keep Pi aligned with the checked-out Lavra source before consulting
    // copied harness skills or the installed npm snapshot.
    const sourceFiles = [
      path.join(lavraPluginsPath(), "skills", name, "SKILL.md"),
      path.join(lavraPackageRoot(), "plugins/lavra/codex/skills", name, "SKILL.md"),
      ...[os.homedir()].flatMap((root) => dirs.map((dir) => path.join(root, dir, name, "SKILL.md"))),
    ];
    return sourceFiles.find((file) => fs.existsSync(file)) ?? null;
  }

  function adaptLoadedSkill(name: string, content: string, cwd?: string, args = ""): string {
    let adapted = content;
    if (name === "lavra-review") {
      // The upstream fallback scans every Lavra agent, including research
      // agents. A work review should use review agents only unless the
      // project explicitly configures a different review_agents list.
      adapted = adapted.replace(
        "**Config-missing behavior:** If `.lavra/config/project-setup.md` absent, dispatch all `DISCOVERED_AGENTS`.",
        "**Config-missing behavior:** If `.lavra/config/project-setup.md` is absent, dispatch only the default review agents from `references/default-agents.md`. Never dispatch agents from research, design, docs, or workflow categories unless explicitly configured.",
      );
      adapted += "\n\n## Pi bridge review limit\nFor ordinary `/lavra-work` reviews, do not dispatch research agents. Research belongs to `/lavra-research` or the planning/design workflow.\n";
    }
    return adaptClaudeMessageText(adapted, cwd, args);
  }

  /**
   * Translate Claude's Skill(...) directives into instructions for Pi's
   * lavra_skill tool. Inlining every skill would create huge prompts and
   * would recursively expand workflow cycles, so skills are loaded on demand.
   */
  function translateSkillDirectives(content: string, commandArgs: string, cwd?: string): string {
    let output = "";
    let cursor = 0;
    let searchFrom = 0;
    const marker = /Skill\s*\(/gi;

    while (true) {
      marker.lastIndex = searchFrom;
      const match = marker.exec(content);
      if (!match) break;

      const openParen = match.index + match[0].length - 1;
      let depth = 1;
      let quote: string | null = null;
      let escaped = false;
      let closeParen = -1;

      for (let i = openParen + 1; i < content.length; i++) {
        const char = content[i];
        if (quote) {
          if (escaped) escaped = false;
          else if (char === "\\") escaped = true;
          else if (char === quote) quote = null;
        } else if (char === "\\" || char === "'" || char === '"') {
          if (char !== "\\") quote = char;
        } else if (char === "(") {
          depth++;
        } else if (char === ")" && --depth === 0) {
          closeParen = i;
          break;
        }
      }

      if (closeParen < 0) break;

      const body = content.slice(openParen + 1, closeParen);
      const nameMatch = body.match(/^\s*(?:["']([^"']+)["']|([a-z0-9][a-z0-9-]*))/i);
      if (!nameMatch) {
        output += content.slice(cursor, closeParen + 1);
        cursor = closeParen + 1;
        searchFrom = cursor;
        continue;
      }

      const name = nameMatch[1] || nameMatch[2];
      if (!skillFilePath(name, cwd)) {
        output += content.slice(cursor, match.index);
        output += `[Pi adaptation: skill "${name}" is unavailable.]`;
      } else {
        const hasExplicitArgs = body.slice(nameMatch[0].length).trim().length > 0;
        const argsText = hasExplicitArgs
          ? " using the arguments specified by this workflow"
          : commandArgs
            ? ` with arguments ${JSON.stringify(commandArgs)}`
            : "";
        output += content.slice(cursor, match.index);
        output += `[Pi adaptation: call the lavra_skill tool with name "${name}"${argsText}. Then follow the returned SKILL.md instructions.]`;
      }
      cursor = closeParen + 1;
      searchFrom = cursor;
    }

    return output + content.slice(cursor);
  }

  /** Dispatch a skill, preferring the user's cross-harness copy when present. */
  function dispatchSkill(name: string, args: string, ctx: ExtensionContext): void {
    const file = skillFilePath(name, ctx.cwd);
    if (!file) {
      ctx.ui.notify(`Lavra skill not found: ${name}`, "error");
      return;
    }

    // Load the selected file directly instead of letting Pi resolve a stale
    // package copy with the same skill name.
    let content = fs.readFileSync(file, "utf-8");
    content = content.replace(/\$ARGUMENTS|#\$ARGUMENTS/g, args || "");
    pi.sendUserMessage(adaptLoadedSkill(name, content, ctx.cwd, args));
  }

  function chooseWorkSkill(args: string, cwd: string): "lavra-work-single" | "lavra-work-multi" {
    const input = args.replace(/--yes\b|--no-parallel\b/g, "").trim();
    if (input.split(",").filter(Boolean).length > 1) return "lavra-work-multi";

    const root = findProjectRoot(cwd);
    try {
      const items = input
        ? JSON.parse(String(execFileSync("bd", ["list", "--parent", input, "--status=open", "--json"], { cwd: root, encoding: "utf-8" })))
        : JSON.parse(String(execFileSync("bd", ["ready", "--json"], { cwd: root, encoding: "utf-8" })));
      return Array.isArray(items) && items.length > 1 ? "lavra-work-multi" : "lavra-work-single";
    } catch {
      return "lavra-work-single";
    }
  }

  function splitCallArguments(body: string): string[] {
    const parts: string[] = [];
    let start = 0;
    let depth = 0;
    let quote: string | null = null;
    let escaped = false;
    for (let i = 0; i < body.length; i++) {
      const char = body[i];
      if (quote) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === quote) quote = null;
      } else if (char === "'" || char === '"') {
        quote = char;
      } else if (char === "(") {
        depth++;
      } else if (char === ")") {
        depth--;
      } else if (char === "," && depth === 0) {
        parts.push(body.slice(start, i).trim());
        start = i + 1;
      }
    }
    parts.push(body.slice(start).trim());
    return parts.filter(Boolean);
  }

  function unquote(value: string): string {
    const trimmed = value.trim();
    if (trimmed.length >= 2 &&
        ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
         (trimmed.startsWith("'") && trimmed.endsWith("'")))) {
      return trimmed.slice(1, -1).replace(/\\(["'\\])/g, "$1");
    }
    return trimmed;
  }

  /** Translate Claude Task(...) calls into Pi lavra_subagent instructions. */
  function translateTaskDirectives(content: string): string {
    let output = "";
    let cursor = 0;
    let searchFrom = 0;
    const marker = /Task\s*\(/gi;

    while (true) {
      marker.lastIndex = searchFrom;
      const match = marker.exec(content);
      if (!match) break;
      const openParen = match.index + match[0].length - 1;
      let depth = 1;
      let quote: string | null = null;
      let escaped = false;
      let closeParen = -1;
      for (let i = openParen + 1; i < content.length; i++) {
        const char = content[i];
        if (quote) {
          if (escaped) escaped = false;
          else if (char === "\\") escaped = true;
          else if (char === quote) quote = null;
        } else if (char === "'" || char === '"') {
          quote = char;
        } else if (char === "(") {
          depth++;
        } else if (char === ")" && --depth === 0) {
          closeParen = i;
          break;
        }
      }
      if (closeParen < 0) break;

      const body = content.slice(openParen + 1, closeParen);
      output += content.slice(cursor, match.index);
      if (/\bteam_name\s*=/.test(body)) {
        output += "[Pi adaptation: this Task is part of the deferred team workflow; TeamCreate/SendMessage support is not enabled yet.]";
      } else {
        const parts = splitCallArguments(body);
        const namedAgent = body.match(/(?:subagent_type|agent)\s*=\s*(["']?)([a-z0-9][a-z0-9-]*)\1/i);
        const agent = namedAgent?.[2] || unquote(parts[0] || "general-purpose");
        const namedModel = body.match(/model\s*=\s*(["']?)([a-z0-9][a-z0-9./:-]*)\1/i);
        const model = namedModel?.[2];
        const namedPrompt = body.match(/prompt\s*=\s*(["'])([\s\S]*)\1\s*$/i);
        let task = namedPrompt?.[2] || "";
        if (!task) {
          for (let i = parts.length - 1; i >= 1; i--) {
            if (/^["']/.test(parts[i])) {
              task = unquote(parts[i]);
              break;
            }
          }
        }
        if (!task) task = parts.slice(1).join(", ") || body;
        output += `[Pi adaptation: call lavra_subagent with agent ${JSON.stringify(agent)}${model ? `, model ${JSON.stringify(model)}` : ""} and task ${JSON.stringify(task)}. Do not use Claude's Task tool.]`;
      }
      cursor = closeParen + 1;
      searchFrom = cursor;
    }
    return output + content.slice(cursor);
  }

  /** Adapt Claude-only directives in expanded skill/user messages. */
  function adaptClaudeMessageText(text: string, cwd?: string, commandArgs = ""): string {
    return translateTaskDirectives(
      translateSkillDirectives(text, commandArgs, cwd)
        .replace(/\bAskUserQuestion\s+tool\b/g, "the `ask_user` tool")
        .replace(/\bAskUserQuestion\b/g, "the `ask_user` tool"),
    );
  }

  /** Read a command .md or skill SKILL.md from @lavralabs/lavra, inject args, send to model */
  function sendCommandFile(name: string, args: string, ctx: ExtensionContext): void {
    // Try command file first, then skill file
    const cmdFile = path.join(lavraPluginsPath(), "commands", `${name}.md`);
    const skillFile = path.join(lavraPluginsPath(), "skills", name, "SKILL.md");
    const file = fs.existsSync(cmdFile) ? cmdFile
      : fs.existsSync(skillFile) ? skillFile
      : null;
    if (!file) {
      ctx.ui.notify(`No command or skill found for "${name}"`, "error");
      return;
    }
    let content = fs.readFileSync(file, "utf-8");
    content = content.replace(/\$ARGUMENTS|#\$ARGUMENTS/g, args || "");
    content = adaptClaudeMessageText(content);
    // Slash commands are invoked while idle; start the translated workflow now.
    pi.sendUserMessage(content);
  }

  // Pi-native replacement for Claude Code's Skill(...) mechanism.
  pi.registerTool({
    name: "lavra_skill",
    label: "Lavra Skill",
    description: "Load and return a Lavra SKILL.md by name. Use this when a Lavra workflow says Skill(\"name\").",
    parameters: Type.Object({
      name: Type.String({ description: "Lavra skill name, for example lavra-work-single" }),
      arguments: Type.Optional(Type.String({ description: "Arguments to pass to the skill" })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const file = skillFilePath(params.name, _ctx.cwd);
      if (!file) {
        return {
          content: [{ type: "text", text: `Lavra skill not found: ${params.name}` }],
          isError: true,
        };
      }
      const skill = adaptLoadedSkill(
        params.name,
        fs.readFileSync(file, "utf-8"),
        _ctx.cwd,
        params.arguments ?? "",
      );
      const args = params.arguments
        ? `\n\nUser arguments for this skill:\n<untrusted-input>${params.arguments}</untrusted-input>`
        : "";
      return { content: [{ type: "text", text: skill + args }] };
    },
  });

  const commands: Array<{
    name: string;
    description: string;
    handler: (args: string, ctx: ExtensionContext) => Promise<void>;
  }> = [
    {
      name: "lavra-work",
      description: "Execute work on one or many beads — auto-routes single/sequential/parallel",
      handler: async (args, ctx) => {
        // Let the current cross-harness lavra-work router decide between
        // single, sequential, and parallel execution. The bundled bridge
        // fallback is retained for installs without an override skill.
        if (skillFilePath("lavra-work", ctx.cwd)) {
          dispatchSkill("lavra-work", args, ctx);
        } else {
          dispatchSkill(chooseWorkSkill(args, ctx.cwd), args, ctx);
        }
      },
    },
    {
      name: "lavra-design",
      description: "Full design pipeline: brainstorm, plan, research, revise, review, lock",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-design", args, ctx);
      },
    },
    {
      name: "lavra-research",
      description: "Gather evidence and best practices using domain-matched research agents",
      handler: async (args, ctx) => {
        dispatchSkill("lavra-research", args, ctx);
      },
    },
    {
      name: "lavra-review",
      description: "Exhaustive code review using multi-agent analysis (4+ review agents)",
      handler: async (args, ctx) => {
        dispatchSkill("lavra-review", args, ctx);
      },
    },
    {
      name: "lavra-plan",
      description: "Create detailed implementation plan from an epic/story bead",
      handler: async (args, ctx) => {
        dispatchSkill("lavra-plan", args, ctx);
      },
    },
    {
      name: "lavra-brainstorm",
      description: "Interactive brainstorming with structured output",
      handler: async (args, ctx) => {
        dispatchSkill("lavra-brainstorm", args, ctx);
      },
    },
    {
      name: "lavra-qa",
      description: "Browser-based QA verification (uses agent-browser)",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-qa", args, ctx);
      },
    },
    {
      name: "lavra-checkpoint",
      description: "Save session progress: file beads, capture knowledge, sync state",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-checkpoint", args, ctx);
      },
    },
    {
      name: "lavra-quick",
      description: "Quick task without full Lavra workflow overhead",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-quick", args, ctx);
      },
    },
    {
      name: "lavra-recall",
      description: "FTS5 full-text search of the knowledge base (SQLite BM25 ranked)",
      handler: async (args, ctx) => {
        if (!args) {
          ctx.ui.notify("Usage: /lavra-recall <query>", "warning");
          return;
        }
        const projectRoot = findProjectRoot(ctx.cwd);
        const results = searchKnowledge(projectRoot, args);
        if (results.length === 0) {
          ctx.ui.notify("No matching knowledge found.", "info");
          return;
        }
        const formatted = results
          .map((r) => `[${r.type.toUpperCase()}] ${r.content}\n  → ${r.bead} | ${r.tags}`)
          .join("\n\n");
        pi.sendUserMessage(
          `Knowledge recall for "${args}":\n\n${formatted}\n\n_(FTS5 BM25-ranked search)_`,
          { deliverAs: "nextTurn" },
        );
      },
    },
    {
      name: "lavra-learn",
      description: "Manually add a knowledge entry: /lavra-learn LEARNED: content",
      handler: async (args, ctx) => {
        const projectRoot = findProjectRoot(ctx.cwd);
        const m = args.match(/^(INVESTIGATION|LEARNED|DECISION|FACT|PATTERN|DEVIATION):\s*(.*)/s);
        if (!m) {
          ctx.ui.notify("Usage: /lavra-learn TYPE: content", "warning");
          return;
        }
        const type = m[1].toLowerCase();
        const content = m[2].slice(0, 2048);
        const slug = content.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60);
        appendKnowledgeJsonl(projectRoot, {
          key: `${type}-${slug}`,
          type,
          content,
          source: "manual",
          tags: [type],
          ts: Math.floor(Date.now() / 1000),
          bead: "manual",
        });
        ctx.ui.notify("Knowledge saved.", "success");
      },
    },
    {
      name: "lavra-ship",
      description: "Ship completed work: git push, bd close, verify",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-ship", args, ctx);
      },
    },
    {
      name: "lavra-retro",
      description: "Session retrospective: review progress, file follow-up beads",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-retro", args, ctx);
      },
    },
    {
      name: "lavra-work-ralph",
      description: "Autonomous retry mode — iterates until completion or budget exhausted",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-work-ralph", args, ctx);
      },
    },
    {
      name: "lavra-work-teams",
      description: "Spawn persistent worker teammates that self-organize through a ready queue",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-work-teams", args, ctx);
      },
    },
    {
      name: "lavra-import",
      description: "Import a markdown plan into beads as an epic with child tasks",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-import", args, ctx);
      },
    },
    {
      name: "lavra-models",
      description: "Choose Pi models for Lavra fast, default, and quality agents",
      handler: async (_args, ctx) => {
        if (!ctx.hasUI) {
          ctx.ui.notify("Run /lavra-models in interactive Pi mode.", "warning");
          return;
        }

        try {
          await ctx.modelRegistry.refresh();
        } catch {
          // Use the currently loaded catalogue if refresh is unavailable.
        }
        const available = ctx.modelRegistry.getAvailable();
        if (available.length === 0) {
          ctx.ui.notify("No authenticated Pi models are available.", "error");
          return;
        }

        const options = [
          "inherit",
          ...available.map((model) => `${model.provider}/${model.id}`),
        ];
        const fast = await ctx.ui.select("Lavra fast-agent model", options);
        if (!fast) return;
        const defaultModel = await ctx.ui.select("Lavra default-agent model", options);
        if (!defaultModel) return;
        const quality = await ctx.ui.select("Lavra quality-agent model", options);
        if (!quality) return;

        writeModelConfig(findProjectRoot(ctx.cwd), {
          fast: fast === "inherit" ? "inherit" : fast,
          default: defaultModel === "inherit" ? "inherit" : defaultModel,
          quality: quality === "inherit" ? "inherit" : quality,
        });
        ctx.ui.notify(
          `Lavra models saved: fast=${fast}, default=${defaultModel}, quality=${quality}`,
          "success",
        );
      },
    },
    {
      name: "lavra-ready",
      description: "Show ready beads count (replaces Claude Code TeammateIdle hook)",
      handler: async (_args, ctx) => {
        try {
          const result = String(execSync("bd ready --json", { encoding: "utf-8", timeout: 5000 }));
          const beads = JSON.parse(result.trim() || "[]");
          ctx.ui.notify(
            beads.length === 0
              ? "No ready beads."
              : `${beads.length} ready bead(s). Run \`bd ready\` to see them.`,
            "info",
          );
        } catch {
          ctx.ui.notify("Could not check beads. Is bd installed?", "warning");
        }
      },
    },
    {
      name: "lavra-setup",
      description: "Initialize Lavra in this project (bd init, provision memory)",
      handler: async (_args, ctx) => {
        ctx.ui.notify("Lavra Setup: initializing project...", "info");
        const projectRoot = findProjectRoot(ctx.cwd);
        // Run provisioning from the Lavra npm package hooks
        try {
          const provisionScript = path.join(HOOKS_DIR, "provision-memory.sh");
          if (fs.existsSync(provisionScript)) {
            execSync(
              `bash -c 'source "${provisionScript}" && provision_memory_dir "${projectRoot}" "${HOOKS_DIR}"'`,
              { stdio: "inherit", timeout: 30000 },
            );
            ctx.ui.notify("Lavra initialized.", "success");
          }
        } catch (err: any) {
          ctx.ui.notify(`Setup error: ${err.message}`, "error");
        }
      },
    },
  ];

  for (const cmd of commands) {
    pi.registerCommand(cmd.name, {
      description: cmd.description,
      handler: async (args, ctx) => {
        if (!isLavraProject(ctx.cwd)) {
          ctx.ui.notify(
            "This project doesn't use beads. Run `/lavra-setup` or `bd init` first.",
            "warning",
          );
          return;
        }
        await cmd.handler(args, ctx);
      },
    });
  }

  // Native /skill expansion loads raw SKILL.md text. Adapt nested Claude
  // directives just before the provider sees that expanded context.
  pi.on("context", async (event, ctx) => {
    const messages = event.messages.map((message: any) => {
      if (message.role !== "user") return message;
      if (typeof message.content === "string") {
        return { ...message, content: adaptClaudeMessageText(message.content, ctx.cwd) };
      }
      if (!Array.isArray(message.content)) return message;
      return {
        ...message,
        content: message.content.map((part: any) =>
          part.type === "text" ? { ...part, text: adaptClaudeMessageText(part.text, ctx.cwd) } : part,
        ),
      };
    });
    return { messages };
  });

  const generalPurposeAgent: AgentDef = {
    name: "general-purpose",
    description: "General-purpose Lavra worker",
    model: "",
    tools: [],
    color: "blue",
    category: "workflow",
    systemPrompt: "You are a general-purpose implementation agent. Complete the delegated task carefully and report concrete results.",
  };

  function resolveAgent(name: string): AgentDef | undefined {
    return name === "general-purpose"
      ? generalPurposeAgent
      : agents.find((agent) => agent.name === name);
  }

  let activeSubagentStore: SubagentSessionStore | undefined;
  let subagentViewerOpen = false;

  async function withSubagentProgress<T>(
    _ctx: ExtensionContext,
    signal: AbortSignal | undefined,
    work: (progress: SubagentProgressSink | undefined, runSignal: AbortSignal | undefined) => Promise<T>,
  ): Promise<T> {
    const store = new SubagentSessionStore();
    activeSubagentStore = store;
    const progress: SubagentProgressSink = {
      start: (label, task) => store.create(label, task),
      update: (session, text) => store.updateStatus(session, text),
      transcript: (session, event) => store.append(session, event),
    };
    return work(progress, signal);
  }

  async function openSubagentViewer(ctx: ExtensionContext): Promise<void> {
    const store = activeSubagentStore;
    if (ctx.mode !== "tui" || !store) {
      if (ctx.hasUI) ctx.ui.notify("No subagent session is available to review.", "info");
      return;
    }
    if (subagentViewerOpen) return;
    subagentViewerOpen = true;
    try {
      await ctx.ui.custom<void>((tui, theme, _keybindings, done) => {
        const unsubscribe = store.subscribe(() => tui.requestRender());
        const close = () => { unsubscribe(); done(); };
        const viewer = new SubagentSessionViewer(store, theme, close, () => tui.requestRender());
        return {
          render: (width: number) => viewer.render(width),
          handleInput: (data: string) => { viewer.handleInput(data); tui.requestRender(); },
          invalidate: () => viewer.invalidate(),
        };
      });
    } finally {
      subagentViewerOpen = false;
    }
  }

  // Ctrl-Shift-R avoids Pi's native prompt-navigation bindings.
  pi.registerShortcut(Key.ctrlShift("r"), {
    description: "Review subagent sessions",
    handler: (ctx) => openSubagentViewer(ctx),
  });

  async function runTrackedAgent(
    agent: AgentDef,
    task: string,
    label: string,
    ctx: ExtensionContext,
    signal: AbortSignal | undefined,
    progress: SubagentProgressSink | undefined,
    modelOverride?: string,
  ) {
    const model = resolveAgentModel(agent, ctx, modelOverride);
    const session = progress?.start(label, task);
    if (session) progress.update(session, `starting (${model ?? "default model"})`);
    const result = await runAgent(
      agent, task, ctx.cwd, signal,
      (text) => session && progress?.update(session, text),
      (event) => session && progress?.transcript(session, event),
      model,
    );
    if (session) progress?.update(session, result.error ? `failed: ${result.error}` : "complete");
    return result;
  }

  // ════════════════════════════════════════════
  //  4. LAVRA_SUBAGENT TOOL
  //     (replaces SubagentStop hook — agents prompted to log learnings)
  // ════════════════════════════════════════════

  if (process.env.LAVRA_PI_SUBAGENT !== "1") pi.registerTool({
    name: "lavra_subagent",
    label: "Lavra Subagent",
    description:
      "Delegate tasks to Lavra's 30 specialized agents with isolated context. " +
      "Each agent runs as a separate pi process with its configured model. " +
      "Modes: single (agent name + task), parallel (array of agents+task), " +
      "chain (sequential steps with {previous} placeholder). " +
      "Agents are loaded from @lavralabs/lavra npm package at " + AGENTS_DIR,
    parameters: Type.Object({
      agent: Type.Optional(Type.String({ description: "Agent name (e.g. security-sentinel, best-practices-researcher)" })),
      model: Type.Optional(Type.String({ description: "Model tier (fast, default, quality) or provider/model override" })),
      task: Type.Optional(Type.String({ description: "Task for the agent" })),

      agents: Type.Optional(Type.Array(
        Type.Object({
          agent: Type.String(),
          model: Type.Optional(Type.String()),
          task: Type.String(),
        }),
        { description: "Parallel tasks (max 6)" },
      )),
      chain: Type.Optional(Type.Array(
        Type.Object({
          agent: Type.String(),
          model: Type.Optional(Type.String()),
          task: Type.String(),
        }),
        { description: "Sequential steps. Use {previous} in task to reference prior output." },
      )),
      capture_learnings: Type.Optional(
        Type.Boolean({
          description: "If true, prompt agent to log LEARNED/DECISION comments before exiting " +
            "(replaces subagent-wrapup.sh hook)",
          default: true,
        }),
      ),
    }),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (agents.length === 0) {
        return {
          content: [{ type: "text", text: "No agents loaded. Is @lavralabs/lavra installed?" }],
          isError: true,
        };
      }

      const captureLearnings = params.capture_learnings ?? true;

      return withSubagentProgress(ctx, signal, async (progress, runSignal) => {
        // ── Single agent mode ──
      if (params.agent && params.task) {
        const agent = resolveAgent(params.agent);
        if (!agent) {
          return {
            content: [{ type: "text", text: `Unknown agent "${params.agent}". Available: ${agents.map((a) => a.name).join(", ")}` }],
            isError: true,
          };
        }
        // Append wrapup prompt if capturing learnings (replaces subagent-wrapup.sh)
        const task = captureLearnings
          ? params.task +
            "\n\nBefore completing, log key learnings using:\n" +
            "bd comments add BEAD_ID \"LEARNED: ...\"\n" +
            "bd comments add BEAD_ID \"DECISION: ...\"\n" +
            "(Replace BEAD_ID with the actual bead ID if applicable.)"
          : params.task;

        const result = await runTrackedAgent(
          agent,
          task,
          agent.name,
          ctx,
          runSignal,
          progress,
          params.model,
        );
        return {
          content: [{ type: "text", text: result.error || boundedAgentOutput(result.output || "(no output)") }],
          isError: !!result.error,
        };
      }

      // ── Parallel mode ──
      if (params.agents && params.agents.length > 0) {
        const results = new Array<string>(params.agents.length);
        let next = 0;
        const worker = async () => {
          while (true) {
            const index = next++;
            if (index >= params.agents!.length) return;
            const task = params.agents![index];
            const agent = resolveAgent(task.agent);
            if (!agent) {
              results[index] = `## ${task.agent}: unknown agent`;
              continue;
            }
            const r = await runTrackedAgent(
              agent,
              task.task,
              `${task.agent} #${index + 1}`,
              ctx,
              runSignal,
              progress,
              task.model,
            );
            results[index] = `## ${task.agent}\n\n${r.error || boundedAgentOutput(r.output || "(no output)")}`;
          }
        };
        const workers = Math.min(maxParallelAgents(findProjectRoot(ctx.cwd)), params.agents.length);
        await Promise.all(Array.from({ length: workers }, () => worker()));
        return {
          content: [{ type: "text", text: results.join("\n\n---\n\n") }],
        };
      }

      // ── Chain mode ──
      if (params.chain && params.chain.length > 0) {
        let previous = "";
        const outputs: string[] = [];
        for (let i = 0; i < params.chain.length; i++) {
          const step = params.chain[i];
          const agent = resolveAgent(step.agent);
          if (!agent) {
            outputs.push(`Step ${i + 1} (${step.agent}): unknown agent`);
            break;
          }
          const task = step.task.replace(/\{previous\}/g, previous);
          const result = await runTrackedAgent(
            agent,
            task,
            `step ${i + 1}: ${step.agent}`,
            ctx,
            runSignal,
            progress,
            step.model,
          );
          if (result.error) {
            outputs.push(`Step ${i + 1} (${step.agent}) failed: ${result.error}`);
            break;
          }
          previous = boundedAgentOutput(result.output);
          outputs.push(`## Step ${i + 1}: ${step.agent}\n\n${boundedAgentOutput(result.output)}`);
        }
        return {
          content: [{ type: "text", text: boundedAgentOutput(outputs.join("\n\n---\n\n")) }],
        };
      }

        return {
          content: [{ type: "text", text: "Provide agent + task, agents[], or chain[]." }],
          isError: true,
        };
      });
    },
  });

  // ════════════════════════════════════════════
  //  5. WEB SEARCH TOOL
  // ════════════════════════════════════════════

  pi.registerTool({
    // Keep Lavra's fallback search tool namespaced so it can coexist with
    // pi-web-access, which owns the standard `web_search` name.
    name: "lavra_web_search",
    label: "Lavra Web Search",
    description: "Search the web for documentation, best practices, and references. " +
      "Uses Brave Search API (BRAVE_API_KEY env var). " +
      "Research agents use this to find current best practices.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      count: Type.Optional(Type.Number({ description: "Results (default 5)" })),
    }),
    promptSnippet: "Web search for documentation and references",
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      return { content: [{ type: "text", text: await webSearch(params.query) }] };
    },
  });

  // ════════════════════════════════════════════
  //  6. FRAMEWORK DOCS TOOL (replaces Context7 MCP)
  // ════════════════════════════════════════════

  pi.registerTool({
    name: "framework_docs",
    label: "Framework Docs",
    description: "Fetch official framework/library documentation via Context7 API. " +
      "Replaces the .mcp.json MCP server with direct HTTP calls.",
    parameters: Type.Object({
      query: Type.String({ description: "Documentation query (e.g. 'Rails Active Storage files')" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      return { content: [{ type: "text", text: await fetchContext7Docs(params.query, params.limit) }] };
    },
  });

  // ════════════════════════════════════════════
  //  7. KNOWLEDGE SEARCH TOOL (FTS5)
  // ════════════════════════════════════════════

  pi.registerTool({
    name: "knowledge_search",
    label: "Knowledge Search",
    description: "Full-text search of Lavra's knowledge base using SQLite FTS5 with BM25 ranking. " +
      "Searches .lavra/memory/knowledge.db. Returns ranked results with type, content, bead, and tags.",
    parameters: Type.Object({
      query: Type.String({ description: "Search terms (will be tokenized)" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!hasSqlite3()) {
        return {
          content: [{ type: "text", text: "sqlite3 CLI not found. Install it for FTS5 knowledge search." }],
        };
      }
      // Ensure DB is synced before search
      try { syncKnowledge(findProjectRoot(ctx.cwd)); } catch { /* proceed even if sync fails */ }
      const results = searchKnowledge(findProjectRoot(ctx.cwd), params.query, params.limit);
      if (results.length === 0) {
        return { content: [{ type: "text", text: "No matching knowledge found." }] };
      }
      const text = results
        .map((r) => `[${r.type.toUpperCase()}] ${r.content}\n  → bead: ${r.bead} | tags: ${r.tags}`)
        .join("\n\n");
      return { content: [{ type: "text", text }] };
    },
  });

  // ════════════════════════════════════════════
  //  8. LIST AGENTS TOOL
  // ════════════════════════════════════════════

  pi.registerTool({
    name: "list_lavra_agents",
    label: "List Lavra Agents",
    description: "List all 30 Lavra agents with descriptions, categories, and configured models.",
    parameters: Type.Object({
      category: Type.Optional(
        Type.String({ description: "Filter: review, research, design, workflow, docs" }),
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const filtered = params.category
        ? agents.filter((a) => a.category === params.category)
        : agents;
      const byCategory = new Map<string, AgentDef[]>();
      for (const a of filtered) {
        byCategory.set(a.category, [...(byCategory.get(a.category) ?? []), a]);
      }
      const text = [...byCategory.entries()]
        .map(([cat, ags]) =>
          `### ${cat} (${ags.length})\n` +
          ags.map((a) => `- **${a.name}** \`[${a.model}]\`: ${a.description}`).join("\n"),
        )
        .join("\n\n");
      return { content: [{ type: "text", text: text || "No agents found." }] };
    },
  });
}
