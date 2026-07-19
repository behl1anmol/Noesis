# Risk register

Risks are numbered stably across revisions; closed risks are kept (struck through), not deleted. Source: `architecture-docs/code-indexer-expanded-architecture.md`, Appendix B (risks 1–8 belong to the approved baseline documents).

| # | Risk | Severity | Mitigation / status |
|---|---|---|---|
| 9 | ~~Reranker latency/memory unacceptable on developer laptops~~ | **Closed (M4)** | Confirmed by measurement: 13.4 s p95 per reranked query even on a T4 GPU at fp32. Resolved by [ADR-35](decisions.md): default-off / per-request opt-in; lazy load + `enabled=false` kill switch keep memory cost at zero unless opted in |
| 10 | ~~Hosted embedder leaks sensitive content off-machine~~ | **Closed (rev 2)** | Eliminated at the source: hosted embedding rejected per [ADR-25](decisions.md). Residual guard: CI grep for HTTP clients in `core/`; dependency-addition rule |
| 11 | ast-grep vs tree-sitter-language-pack language-name drift | Low | Single explicit `LANGUAGE_MAP`; clean unsupported-language error |
| 12 | ~~Git fast-path silently wrong after history rewrite~~ | **Closed (M7)** | Anchor validity is `git merge-base --is-ancestor <anchor> HEAD` (a rewritten-away anchor fails ancestry even before gc) → silent full-walk fallback; hash still confirms every candidate; every fallback condition has its own test asserting identical final partitioning ([ADR-37](decisions.md)) |
| 13 | MCP v2 stable (2026-07-28) breaks transports mid-build | High | Scheduled [M9 checkpoint](milestones.md); hard pins (`mcp<2`); FastMCP owns the `mcp` range; transport integration tests as the tripwire |
| 14 | Scope expansion erodes the M3 evaluation discipline | Medium | The gate is a named milestone exit criterion; M4 explicitly blocked on it; the `/eval` harness makes re-measurement one keystroke |
| 15 | Lesson store degrades: context bloat, stale lessons, or a persisted rationalization that weakens discipline | Medium | 15-lesson hard cap forcing promote-or-retire; lessons subordinate to the hard rules by construction; committed `LESSONS.md` gets PR review; human-only promotion to permanent rules |

See [Lessons](lessons.md) for how risk 15's guardrails work in practice.
