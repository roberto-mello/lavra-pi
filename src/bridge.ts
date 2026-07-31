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

import { spawn, execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { StringEnum } from "@earendil-works/pi-ai";
import {
  type ExtensionAPI,
  type ExtensionContext,
  isToolCallEventType,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// ═══════════════════════════════════════════════════
// Path resolution from @lavralabs/lavra npm package
// ═══════════════════════════════════════════════════

/** Resolve the lavra package root inside node_modules */
function lavraPackageRoot(): string {
  // When pi installs a git package and runs npm install, @lavralabs/lavra
  // ends up in node_modules/@lavralabs/lavra/ relative to this package root.
  const candidates = [
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
        model: fm.model ?? "sonnet",
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

async function runAgent(
  agent: AgentDef,
  task: string,
  cwd: string,
  signal?: AbortSignal,
  onProgress?: (text: string) => void,
): Promise<{ output: string; error?: string; usage: any }> {
  const args: string[] = ["--mode", "json", "-p", "--no-session"];
  if (agent.model) args.push("--model", agent.model);
  if (agent.tools && agent.tools.length > 0) {
    args.push("--tools", agent.tools.join(","));
  }

  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "lavra-agent-"));
  const promptFile = path.join(tmpDir, "prompt.md");
  try {
    const fullPrompt = `${agent.systemPrompt}\n\nTask: ${task}`;
    await fs.promises.writeFile(promptFile, fullPrompt, { encoding: "utf-8", mode: 0o600 });
    args.push(promptFile);
    args.push("(continue)");
  } catch {
    // fallback: pass task inline
    args.push(`Task: ${task}`);
  }

  return new Promise((resolve) => {
    const proc = spawn("pi", args, {
      cwd,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
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
          if (evt.type === "message_end" && evt.message) {
            messages.push(evt.message);
            for (const part of evt.message.content ?? []) {
              if (part.type === "text") {
                stdout = part.text;
                onProgress?.(part.text);
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
    pi.sendUserMessage(content, { deliverAs: "nextTurn" });
  }

  const commands: Array<{
    name: string;
    description: string;
    handler: (args: string, ctx: ExtensionContext) => Promise<void>;
  }> = [
    {
      name: "lavra-work",
      description: "Execute work on one or many beads — auto-routes single/sequential/parallel",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-work", args, ctx);
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
        sendCommandFile("lavra-research", args, ctx);
      },
    },
    {
      name: "lavra-review",
      description: "Exhaustive code review using multi-agent analysis (4+ review agents)",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-review", args, ctx);
      },
    },
    {
      name: "lavra-plan",
      description: "Create detailed implementation plan from an epic/story bead",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-plan", args, ctx);
      },
    },
    {
      name: "lavra-brainstorm",
      description: "Interactive brainstorming with structured output",
      handler: async (args, ctx) => {
        sendCommandFile("lavra-brainstorm", args, ctx);
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
        // Run provisioning from vendored hooks
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

  // ════════════════════════════════════════════
  //  4. LAVRA_SUBAGENT TOOL
  //     (replaces SubagentStop hook — agents prompted to log learnings)
  // ════════════════════════════════════════════

  pi.registerTool({
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
      task: Type.Optional(Type.String({ description: "Task for the agent" })),
      agents: Type.Optional(Type.Array(
        Type.Object({
          agent: Type.String(),
          task: Type.String(),
        }),
        { description: "Parallel tasks (max 6)" },
      )),
      chain: Type.Optional(Type.Array(
        Type.Object({
          agent: Type.String(),
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

      // ── Single agent mode ──
      if (params.agent && params.task) {
        const agent = agents.find((a) => a.name === params.agent);
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

        const result = await runAgent(agent, task, ctx.cwd, ctx.signal);
        return {
          content: [{ type: "text", text: result.output || result.error || "(no output)" }],
          isError: !!result.error,
        };
      }

      // ── Parallel mode ──
      if (params.agents && params.agents.length > 0) {
        const results = await Promise.all(
          params.agents.map(async (a) => {
            const agent = agents.find((ag) => ag.name === a.agent);
            if (!agent) return `## ${a.agent}: unknown agent`;
            const r = await runAgent(agent, a.task, ctx.cwd, ctx.signal);
            return `## ${a.agent}\n\n${r.output || r.error || "(no output)"}`;
          }),
        );
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
          const agent = agents.find((a) => a.name === step.agent);
          if (!agent) {
            outputs.push(`Step ${i + 1} (${step.agent}): unknown agent`);
            break;
          }
          const task = step.task.replace(/\{previous\}/g, previous);
          const result = await runAgent(agent, task, ctx.cwd, ctx.signal);
          if (result.error) {
            outputs.push(`Step ${i + 1} (${step.agent}) failed: ${result.error}`);
            break;
          }
          previous = result.output;
          outputs.push(`## Step ${i + 1}: ${step.agent}\n\n${result.output}`);
        }
        return {
          content: [{ type: "text", text: outputs.join("\n\n---\n\n") }],
        };
      }

      return {
        content: [{ type: "text", text: "Provide agent + task, agents[], or chain[]." }],
        isError: true,
      };
    },
  });

  // ════════════════════════════════════════════
  //  5. WEB SEARCH TOOL
  // ════════════════════════════════════════════

  pi.registerTool({
    name: "web_search",
    label: "Web Search",
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
