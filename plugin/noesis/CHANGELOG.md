# Changelog

All notable changes to the Noesis plugin are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.1.0] - 2026-07-06

### Added
- Initial release.
- MCP connection to a running Noesis service over HTTP (`.mcp.json`), with a
  configurable `base_url` (default `http://127.0.0.1:8000`).
- `noesis-mcp` skill:
  - `SKILL.md` operational guide (golden path, tool quick reference, registration
    and connection recovery).
  - `references/tools.md`, `references/workflows.md`, `references/transports.md`,
    `references/troubleshooting.md`.
  - `scripts/register_project.py` (register a repo via REST, optional `--wait`).
  - `scripts/healthcheck.py` (diagnose service connectivity).
