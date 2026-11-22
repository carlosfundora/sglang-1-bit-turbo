# PREFERENCES

**Version:** 1.0.0  
**Last Updated:** 2026-05-19  
**Status:** Active

## Purpose

This is a self-contained, project-agnostic guide to preferred engineering standards, policies, and development style.

---

## Strict Standards (MUST / NEVER)

### Canonicality and ownership

1. **MUST** maintain one canonical source per policy domain; **NEVER** keep multiple mutable canonicals.
2. **MUST** treat duplicates as distribution artifacts; **NEVER** edit replicas as authority.
3. **MUST** define explicit ownership for standards domains; **NEVER** rely on implicit stewardship.
4. **MUST** use scripted sync/export for replicated artifacts; **NEVER** rely on ad-hoc manual copy flow.
5. **MUST** use deterministic migration paths when canonical locations change; **NEVER** leave mixed-state ambiguity.
6. **MUST** preserve policy change auditability; **NEVER** perform silent governance edits.
7. **MUST** state scope/applicability in standards docs; **NEVER** publish context-free rules.

### Documentation governance

8. **MUST** enforce deterministic file placement by topic/domain; **NEVER** use roots as dumping grounds.
9. **MUST** use explicit descriptive naming (lowercase kebab-case where applicable); **NEVER** use vague names like `misc`/`temp`.
10. **MUST** keep one topic per canonical document; **NEVER** split a single policy topic across overlapping files.
11. **MUST** consolidate semantic duplicates; **NEVER** allow indefinite parallel policy copies.
12. **MUST** define exceptions explicitly; **NEVER** allow undocumented special cases.
13. **MUST** enforce structural limits in high-noise directories; **NEVER** allow unbounded doc sprawl.
14. **MUST** keep canonical filenames stable; **NEVER** encode transient state in permanent policy filenames.
15. **MUST** separate canonical standards from temporary notes; **NEVER** elevate ephemeral logs to policy by default.

### Metadata and machine-parsability

16. **MUST** require complete metadata/front matter where format requires it; **NEVER** permit missing identity fields.
17. **MUST** place required metadata in discovery-safe position; **NEVER** prepend content that breaks parsers.
18. **MUST** optimize standards for both human and agent consumption; **NEVER** optimize exclusively for one.
19. **MUST** keep hard-rule language explicit; **NEVER** blur mandatory vs optional wording.
20. **MUST** separate normative rules from guidance text; **NEVER** mix policy and preference in one statement.

### Environment and dependency management

21. **MUST** standardize one primary environment toolchain for reproducibility; **NEVER** mix primary package managers.
22. **MUST** use lockfiles for deterministic rebuilds; **NEVER** run critical workflows from unconstrained installs.
23. **MUST** pin index/channel sources where hardware/runtime compatibility matters; **NEVER** allow resolver drift.
24. **MUST** enforce hardware-target compatibility in dependency policy; **NEVER** allow incompatible runtime-family packages.
25. **MUST** use reproducible environment creation patterns; **NEVER** allow unmanaged snowflake envs in canonical paths.
26. **MUST** promote only reviewed workflows to canonical process; **NEVER** canonize convenience hacks directly.

### Service registration, ports, and runtime wiring

27. **MUST** use central service registry as runtime truth; **NEVER** treat hardcoded local values as authority.
28. **MUST** follow explicit naming/domain prefix rules for services; **NEVER** register ambiguous service identities.
29. **MUST** allocate ports from approved domain blocks; **NEVER** assign ports from forbidden ranges.
30. **MUST** resolve runtime ports through generated environment mappings; **NEVER** hardcode service ports.
31. **MUST** run conflict checks before registration changes; **NEVER** assume capacity by convention.
32. **MUST** update all dependent layers after registry changes; **NEVER** partially update one layer.
33. **MUST** enforce service-appropriate health depth; **NEVER** claim readiness from superficial liveness alone.
34. **MUST** isolate secrets from shared runtime env maps; **NEVER** put tokens/keys in public env mapping files.
35. **MUST** keep service-plane and client-plane semantics explicit; **NEVER** conflate accessibility and operability.

### Skills and agent operations

36. **MUST** maintain one canonical skills source; **NEVER** run multi-canonical skill mutation.
37. **MUST** validate canonical skill metadata before export; **NEVER** distribute malformed skill definitions.
38. **MUST** export platform skills as decoupled snapshots; **NEVER** runtime-link exports back to canonical source.
39. **MUST** keep target-platform transforms deterministic and script-driven; **NEVER** rely on implicit compatibility.
40. **MUST** keep non-skill/support clutter out of skill roots unless standardized; **NEVER** mix loose pseudo-skills.
41. **MUST** keep naming and activation semantics consistent across platforms; **NEVER** allow silent alias drift.

### Rust catalog governance

42. **MUST** treat the Rust catalog as an aggregate source-of-materials repository; **NEVER** allow it to drift behind improvements from active repos.
43. **MUST** index every catalog crate in catalog inventory/index structures; **NEVER** leave crates unindexed.
44. **MUST** use `rs_` prefix for catalog crate names by default; **NEVER** introduce new non-`rs_` crate names without explicit exception.
45. **MUST** preserve catalog crates by default; **NEVER** hard-delete catalog crates as routine cleanup.
46. **MUST** move obsolete crates to a deprecated path (`rust/deprecated/`) when retiring; **NEVER** remove retired crates without deprecation move.
47. **MUST** aggregate validated upgrades/fixes from sibling repos into catalog materials; **NEVER** keep known improvements siloed long-term.
48. **MUST** keep infrastructure/service client crates under the `clients/` domain directory; **NEVER** scatter infra clients across unrelated crate folders.
49. **MUST** keep workspace wiring separate from catalog storage intent; **NEVER** assume all catalog crates are active workspace members.
50. **MUST** prefer non-destructive sync when reconciling catalog copies; **NEVER** delete during alignment when move/update can preserve history.

### Delivery discipline

51. **MUST** search/reuse before creating new standards artifacts; **NEVER** duplicate due to discovery omission.
52. **MUST** prefer non-destructive catalog alignment by default; **NEVER** delete canonical materials without explicit policy path.
53. **MUST** codify repeat workflows in scripts; **NEVER** depend on tribal memory for critical operations.
54. **MUST** prioritize deterministic, auditable operational behavior; **NEVER** normalize one-off success as standard.

---

## Less-Binary Preference Guidelines (Inferred)

These are preferred tendencies, not hard law.

### Architecture and repo shape

1. Prefer clear domain partitioning over mixed-purpose directories.
2. Prefer explicit boundaries/contracts between modules and services.
3. Prefer small canonical hubs with specialized leaf docs.
4. Prefer policy docs separated from implementation run notes.
5. Prefer runtime topology documentation that mirrors actual deployment wiring.

### Documentation style

6. Prefer concise governance docs with high signal and low narrative overhead.
7. Prefer decision-tree or placement-matrix patterns for routing content.
8. Prefer anti-pattern examples to reduce interpretation ambiguity.
9. Prefer consolidation over proliferation when topics overlap.
10. Prefer update/version context in long-lived governance docs.

### Diagram and visualization preferences

11. Prefer diagrams for system wiring, dependency flow, and boundary clarity.
12. Prefer simple readable flow diagrams over deeply nested complexity.
13. Prefer architecture diagrams in canonical docs, not scattered transient notes.
14. Prefer text-first policy with diagrams as clarifiers.
15. Prefer stable diagram conventions across repositories.

### Runtime and operations working style

16. Prefer generated config layers over repeated manual values.
17. Prefer env-variable wiring for runtime consistency.
18. Prefer explicit constraint zones (allowed/forbidden ranges).
19. Prefer health checks that prove real usability, not just liveness.
20. Prefer strict secret/config separation in operational files.

### Skills and platform packaging

21. Prefer canonical authoring plus platform-target export transforms.
22. Prefer decoupled exported bundles with no runtime dependency on canonical source.
23. Prefer metadata completeness as an export gate.
24. Prefer deterministic export automation over one-off migration scripts.

### Execution and collaboration

25. Prefer hard-cut decisions when ambiguity creates maintenance cost.
26. Prefer explicit defaults over inferred assumptions.
27. Prefer reproducibility and traceability over local speed hacks.
28. Prefer portable standards phrased project-agnostically by default.
