@AGENTS.md

## Claude-specific notes

- This file is loaded into every Claude Code session in this repo, so keep it to
  the import above and a few lines. The project guidance lives in `AGENTS.md`,
  which every agent tool reads.
- For history and rationale, ask the tracker (`gh issue view N`,
  `gh pr view N --comments`) or `git log -S<symbol>`. Do not paste their state
  into either file.
- When `@claude` runs in CI (`.github/workflows/claude.yml`), the same house rules
  apply to its PRs and comments.
