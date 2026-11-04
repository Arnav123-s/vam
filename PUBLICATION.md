# Publication scope and provenance

Prepared 2026-11-04 as a separate public VAM research snapshot. The original
research repository is unchanged and remains separate.

## Included source revisions

- Root package: the complete `vam/` subtree of source revision
  `b350cf647a39b094e6b179a49512f7628e562ac6`, branch `claude/vam-nfa-continue`
  (180 source files).
- `archive/original-branch/`: the complete `vam/` subtree of revision
  `2b4433bb1856f46f20d1457c8ba77e3fe05b22aa`, branch
  `claude/project-thread-20ve0y` (108 source files).

The source is the owner's separate research archive. Revision and blob hashes
are provided for provenance; old commit references in research notes refer to
that source history, not commits in this new repository. The complete original
Git history is not imported because it includes unrelated private research.

Every downloaded source file was checked against its original Git blob SHA-1.
`provenance/source-files.json` maps source paths, revisions and blob hashes to
published paths and SHA-256 hashes, including any publication-only edits.

## Publication-only changes

- Added the Apache-2.0 license, NOTICE and explicit third-party rights notices.
- Added a current README, reproduction guide and this provenance guide.
- Retained the original continuation README as `RESEARCH_README.md`.
- Removed private session identifiers from historical notes and normalized
  machine-specific example paths. The mathematical specifications, executable
  research code, tests, recorded results and E5 question hashes are unchanged.
- Marked handoff and planner documents as historical workflow records.
- Added package license/readme metadata and ignore rules for local secrets,
  generated caches and environments. The eight E5 metadata files and historical
  partial-result evidence are intentionally included.

## Deliberately outside scope

Unrelated SERA workspaces, raw backup archives, account transcripts, Git storage,
installed environments, credentials and local machine configuration are not
included. Full third-party books are restored from their recorded sources rather
than bundled. The archive's partial E3 result remains labeled partial; no missing
run or uncommitted historical change has been invented.

## Interpretation

The root package is the continuation snapshot. The historical branch contains
divergent Page-sizing and audit work. These are retained as an independent
snapshot instead of silently merging mathematical or implementation changes.
Its test receipts and results describe its own historical revision.

The saved FAIL receipt is retained. Publication does not promote failed gates to
passed, claim full current verification, or claim that E8/E9 were completed.
