# lavra-pi

[Lavra](https://lavra.dev) for [pi](https://pi.dev) — beads-based persistent memory,
30 specialized agents, and multi-agent workflows.

## Install

```bash
pi install git:github.com/roberto-mello/lavra-pi
```

**One command. Zero bootstrap. Everything auto-loads.**

The bridge extension resolves agents from `@lavralabs/lavra` (the same npm
package you already publish on `npmjs.com`). When pi runs `npm install`
during the git install, it pulls `@lavralabs/lavra` into `node_modules/`.
The extension discovers agents from there at startup.

## What You Get

| Feature | How It Works |
|---|---|
| **30 agents** (review, research, design, workflow, docs) | Loaded from `node_modules/@lavralabs/lavra/plugins/lavra/agents/` |
| **15+ skills** (SKILL.md) | Agents discover them at runtime by globbing `**/**/SKILL.md` during research — not pre-loaded by pi |
| **~17 commands** (`/lavra-work`, `/lavra-design`, etc.) | `pi.registerCommand()` — registered at startup |
| **Auto-recall** (session start knowledge injection) | `session_start` event handler reads knowledge.jsonl + session state |
| **Memory capture** (post-tool knowledge extraction) | `tool_result` event intercepts `bd comments add` |
| **FTS5 knowledge search** | SQLite FTS5 with BM25 ranking (same algorithm as `knowledge-db.sh`) |
| **Web search** | `web_search` tool — Brave API or agent-browser fallback |
| **Framework docs** (Context7) | `framework_docs` tool — direct `fetch()`, no MCP server |
| **Subagents** | Custom `lavra_subagent` tool — single/parallel/chain modes; Claude `Task(...)` is translated to it |
| **User questions** | Bundled `pi-ask-user` package provides the `ask_user` tool for `AskUserQuestion` workflows |
| **Subagent wrapup** (log learnings before exit) | Built into subagent tool — prompts `LEARNED:`/`DECISION:` comments |
| **Model routing per agent** | Agent `model:` frontmatter → `--model` flag to subagent pi process |

### Skill Dispatch and Conflicts

Lavra skills are registered with Pi and use Pi's native syntax:

```text
/skill:lavra-work-single medsimples-lj5z
```

The bridge routes direct skill commands through `/skill:name`. Nested Claude
`Skill(...)` references are translated to the compatibility `lavra_skill` tool
because a model cannot invoke a slash command from inside another prompt.

Duplicate copies are excluded for `agent-browser` and `frontend-design`, so
user-installed versions win without collision warnings.

## Architecture

```
lavra-pi/
├── package.json              # pi package — declares extension only
├── src/
│   └── bridge.ts             # The whole runtime (~900 lines)
└── node_modules/
    └── @lavralabs/
        └── lavra/            # Resolved at npm install time
            └── plugins/lavra/
                ├── agents/   # 30 .md agent definitions
                ├── skills/   # Not loaded by pi — agents discover at runtime
                └── hooks/    # Available for pi.exec() but not required
```

### No vendoring

`@lavralabs/lavra` is a regular `dependencies` entry. The pi package
does not fork or mirror the Lavra repo. Users always get the version
pinned in `package.json`, and they update with:

```bash
pi update git:github.com/roberto-mello/lavra-pi
```

### Knowledge Search (FTS5)

The memory capture and recall system matches the existing Lavra
`knowledge-db.sh` implementation:

| Operation | Implementation |
|---|---|
| **Capture** | `tool_result` event → parse `bd comments add` → append to `knowledge.jsonl` + `knowledge.db` (SQLite) |
| **Sync** | Incremental: new JSONL lines imported into SQLite FTS5 on each `knowledge_search` tool call |
| **Backfill** | First-time: imports `LEARNED:`/`DECISION:` comments from beads via `bd sql` |
| **Search** | FTS5 `MATCH` with BM25 ranking (weights: content=-10, tags=-5, type=-2, key=-1) |
| **Rotation** | JSONL rotated at 5000 lines (2500 archived) — matches `memory-capture.sh` |
| **CLI** | `/lavra-recall <query>` or `knowledge_search` tool (both use FTS5) |

Requires `sqlite3` CLI (already required by Lavra).

## Hook Replacement Map

```
Claude Code Hook     →  pi Extension Event
─────────────────────────────────────────────────
SessionStart         →  session_start
PostToolUse (Bash)   →  tool_result (on bash tool)
Skill(...)            →  native /skill:name; nested refs → lavra_skill
Task(...)             →  lavra_subagent
AskUserQuestion      →  ask_user (bundled pi-ask-user)
SubagentStop         →  built into lavra_subagent tool lifecycle
TeammateIdle         →  /lavra-ready command
.mcp.json (Context7) →  framework_docs tool (HTTP fetch)
```

## Requirements

- Node.js >= 18
- `bd` (beads) installed for full Lavra functionality
- `sqlite3` CLI (for FTS5 knowledge search)
- Optional: `BRAVE_API_KEY` env var for web search
