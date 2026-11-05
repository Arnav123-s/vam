# Phase 4 (H2) result -- 2026-11-04, logged honestly

## What works
- Governor machinery functions: purge, radiate, restore, entropy monitor all fire.
- Purge >> static under regime cycling (0.820 vs 0.898 total bpb): slot management
  matters; contamination is the real enemy.
- Entropy monitor: S/C 11.9/12.0 -> 10.3/12.0, stays under capacity.

## Null result (robust across 3 variants)
- Radiate/restore ~= plain slot reuse: gaps 0.0002 (20k segs), 0.001 (5k segs),
  0.0013 (rich order-3 structure, 8k segs). The distinctive H2 action shows no
  measurable win HERE.

## Why (the interpretation, not just the number)
Relearning from scratch is cheap **when the slot structure (micro partition) is
already right** -- rebuilding counts takes ~200-1000 bits. The expensive part of
learning is discovering STRUCTURE, not filling parameters. The radiated invariant
worth keeping must be STRUCTURAL (which partition, which factors, which
disentanglers), not a count table.

## Reframing
H2 x H4 fusion: the governor should radiate learned *structure*. A count table is
cheap to rebuild; a discovered factorization is not. Next iteration: radiate the
H4 factors / micro-partition, not the counts.
