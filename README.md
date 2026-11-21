# Sokoban solver — make.fun "Build the best Sokoban solver, 21 November 2026"

One program, `main.ts` (Node 22, no dependencies), that answers all 5
instances of a set: it reads `/input/<id>/level.txt` for each instance and
writes the player's moves to `/output/<id>/solution.txt`.

## How it works

Every level is a hallway plus 1–4 rooms off it, each room holding 3–4 boxes
and as many goals, and a level's push-optimal count is the sum of its rooms'
optima — a box pushed out of a room can never come back, so boxes stay in
their room. The solver therefore:

1. Splits the level into the hallway row and its rooms (each room's door and
   the hallway cell it opens onto are found by flood fill).
2. Solves each room independently and push-optimally with A* over **pushes**
   (a state is the player's square plus the boxes' squares; a push is one
   macro move, with the player's walk to the push found by flood fill).
   The estimate is the minimum-cost matching of boxes to goals by grid
   distance, and dead states are pruned: boxes on cells no goal can be
   reached from (reverse push reachability), corner and frozen boxes, and
   filled 2×2 blocks. A room's solution ends with the player back out at
   the room's entry cell, so the next room can be reached.
3. Stitches the rooms together by walking the player through the hallway
   between doors and concatenating the move strings.

If a room search fails, the level falls back to the same push A* run over
the whole level; if that fails too, the level gets an empty (valid, worth
0) answer rather than no answer. The search is budgeted by expansions,
never by the wall clock, so two executions write the same bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

This README is descriptive and stays local only; the repository README is
built from the make.fun template.
