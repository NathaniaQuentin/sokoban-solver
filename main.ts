// Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
//
// Approach: every level is a hallway plus 1-4 rooms off it, each room holding
// 3-4 boxes and as many goals, with a level's push-optimal count P* the sum of
// its rooms' optima (a box pushed out of a room can never come back, so boxes
// stay in their room). The solver therefore solves each room independently
// and push-optimally, then stitches the rooms together by walking the player
// through the hallway between doors and concatenating the move strings.
//
// Each room is solved by A* over PUSHES (a state is the player's square plus
// the boxes' squares; a move is one push, with the player's walk to the push
// found by flood fill), not over single steps: that is what the score counts.
// The estimate is the minimum-cost matching of boxes to goals by grid
// distance (admissible: every push moves one box one square), and states are
// pruned by deadlock detection: boxes on cells no goal can be reached from
// (reverse push reachability), corner and frozen boxes, and filled 2x2
// blocks. The search is budgeted by expansions, never by the wall clock, so
// two executions write the same bytes. If a room search fails, the level
// falls back to the same push A* run over the whole level; if that fails too
// the level gets an empty (valid, worth 0) answer rather than no answer.
//
// Original work: written for this bounty, not copied from the house baseline
// (a plain move A* with no deadlock knowledge) or from any public solver.
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions: 0 up, 1 down, 2 left, 3 right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];

// Expansion budgets (work, never the clock): per room, for the whole-level
// fallback, and across the whole run. They bound the 60 s wall budget.
const ROOM_CAP = 200_000;
const FALLBACK_CAP = 200_000;
const GLOBAL_CAP = 2_000_000;
let globalExpansions = 0;

// ---------- level parsing ----------

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: boolean[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const raw = text.replace(/\r/g, '').split('\n');
  while (raw.length > 0 && raw[raw.length - 1] === '') raw.pop();
  const h = raw.length;
  const w = Math.max(...raw.map((r) => r.length));
  const wall = new Array<boolean>(w * h).fill(false);
  const goalSet = new Array<boolean>(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = raw[y]?.[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet[i] = true;
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// ---------- room extraction ----------

interface Room {
  cells: number[]; // global cells of the room component, sorted
  boxes: number[];
  goals: number[];
  door: number; // global cell of the room's door
  entry: number; // global hallway cell the door opens onto
}

// Split the level into the hallway row (the player's row) and the rooms off
// it. Returns null when the level does not have that shape.
function extractRooms(lv: Level): Room[] | null {
  const { w, h, wall, player } = lv;
  const hy = Math.floor(player / w);
  const seen = new Array<boolean>(w * h).fill(false);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || Math.floor(i / w) === hy || seen[i]) continue;
    const comp: number[] = [];
    const stack = [i];
    seen[i] = true;
    while (stack.length > 0) {
      const c = stack.pop()!;
      comp.push(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || Math.floor(n / w) === hy || seen[n]) continue;
        seen[n] = true;
        stack.push(n);
      }
    }
    comp.sort((a, b) => a - b);
    const set = new Set(comp);
    const boxes = lv.boxes.filter((b) => set.has(b));
    if (boxes.length === 0) continue; // empty floor pocket, not a room
    const goals = lv.goals.filter((g) => set.has(g));
    let door = -1;
    let entry = -1;
    for (const c of comp) {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (ny === hy && nx >= 0 && nx < w && !wall[ny * w + nx]) {
          door = c;
          entry = ny * w + nx;
        }
      }
    }
    if (door < 0) return null;
    rooms.push({ cells: comp, boxes, goals, door, entry });
  }
  const inRoom = new Set(rooms.flatMap((r) => r.cells));
  if (!lv.boxes.every((b) => inRoom.has(b))) return null;
  if (!lv.goals.every((g) => inRoom.has(g))) return null;
  return rooms;
}

// ---------- search space ----------

interface Space {
  w: number;
  h: number;
  n: number;
  gx: number[];
  gy: number[];
  localOf: Int32Array; // global cell -> local index, -1 when outside the space
  walkable: boolean[]; // a cell the player may stand on
  nbr: Int32Array; // n*4: neighbour local index, -1 when not walkable
  dest: Int32Array; // n*4: push destination local index, -1 when a box may not go there
  goalList: number[];
  isGoal: boolean[];
  dist: number[][]; // dist[cell][goal index]: grid distance ignoring boxes
  live: boolean[]; // a box here can still reach a goal by pushes
  boxes0: number[];
  player0: number;
  entryLocal: number; // the hallway cell a room opens onto, -1 for the whole level
  keyFn: (player: number, boxes: number[]) => number | string;
}

function bfsDistances(space: Space): number[][] {
  const dist: number[][] = [];
  for (let g = 0; g < space.n; g++) dist.push(new Array<number>(space.goalList.length).fill(Infinity));
  for (let j = 0; j < space.goalList.length; j++) {
    const start = space.goalList[j]!;
    const d = dist[start]!;
    d[j] = 0;
    const q = [start];
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      for (let dir = 0; dir < 4; dir++) {
        const nb = space.nbr[c * 4 + dir];
        if (nb < 0 || dist[nb]![j] !== Infinity) continue;
        dist[nb]![j] = d[j]! + 1;
        q.push(nb);
      }
    }
  }
  return dist;
}

// Reverse push reachability: a cell is live when a box there can be pushed
// to a goal. Start from the goals; a cell joins when some push moves a box
// from it onto a live cell and the player can stand on the far side.
function computeLive(space: Space): boolean[] {
  const live = new Array<boolean>(space.n).fill(false);
  for (const g of space.goalList) live[g] = true;
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < space.n; c++) {
      if (live[c]) continue;
      for (let d = 0; d < 4; d++) {
        const to = space.dest[c * 4 + d];
        const from = space.nbr[c * 4 + OPP[d]];
        if (to >= 0 && live[to] && from >= 0) {
          live[c] = true;
          changed = true;
          break;
        }
      }
    }
  }
  return live;
}

function finishSpace(space: Space): Space {
  space.dist = bfsDistances(space);
  space.live = computeLive(space);
  return space;
}

function buildRoomSpace(lv: Level, room: Room): Space {
  const cells = room.cells;
  const entryLocal = cells.length;
  const n = cells.length + 1;
  const localOf = new Int32Array(lv.w * lv.h).fill(-1);
  cells.forEach((g, i) => {
    localOf[g] = i;
  });
  localOf[room.entry] = entryLocal;
  const gx = new Array<number>(n);
  const gy = new Array<number>(n);
  cells.forEach((g, i) => {
    gx[i] = g % lv.w;
    gy[i] = Math.floor(g / lv.w);
  });
  gx[entryLocal] = room.entry % lv.w;
  gy[entryLocal] = Math.floor(room.entry / lv.w);
  const walkable = new Array<boolean>(n).fill(true);
  const nbr = new Int32Array(n * 4).fill(-1);
  const dest = new Int32Array(n * 4).fill(-1);
  for (let c = 0; c < n; c++) {
    for (let d = 0; d < 4; d++) {
      const nx = gx[c]! + DX[d];
      const ny = gy[c]! + DY[d];
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const l = localOf[ny * lv.w + nx];
      if (l < 0) continue;
      nbr[c * 4 + d] = l;
      if (l < cells.length) dest[c * 4 + d] = l; // a box may only stay in the room
    }
  }
  // The player enters and leaves through the door alone: the entry cell's
  // only neighbour inside the search is the door.
  for (let d = 0; d < 4; d++) {
    const nb = nbr[entryLocal * 4 + d];
    if (nb >= 0 && nb !== localOf[room.door]) nbr[entryLocal * 4 + d] = -1;
  }
  const goalList = room.goals.map((g) => localOf[g]!);
  const isGoal = new Array<boolean>(n).fill(false);
  for (const g of goalList) isGoal[g] = true;
  const boxes0 = room.boxes.map((b) => localOf[b]!).sort((a, b) => a - b);
  const space: Space = {
    w: lv.w,
    h: lv.h,
    n,
    gx,
    gy,
    localOf,
    walkable,
    nbr,
    dest,
    goalList,
    isGoal,
    dist: [],
    live: [],
    boxes0,
    player0: entryLocal,
    entryLocal,
    keyFn: (player, boxes) => {
      let k = player;
      for (let i = 0; i < 4; i++) k |= (i < boxes.length ? boxes[i]! : 63) << (6 + 6 * i);
      return k;
    },
  };
  return finishSpace(space);
}

function buildWholeSpace(lv: Level): Space {
  const cells: number[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) cells.push(i);
  const n = cells.length;
  const localOf = new Int32Array(lv.w * lv.h).fill(-1);
  cells.forEach((g, i) => {
    localOf[g] = i;
  });
  const gx = cells.map((g) => g % lv.w);
  const gy = cells.map((g) => Math.floor(g / lv.w));
  const walkable = new Array<boolean>(n).fill(true);
  const nbr = new Int32Array(n * 4).fill(-1);
  const dest = new Int32Array(n * 4).fill(-1);
  for (let c = 0; c < n; c++) {
    for (let d = 0; d < 4; d++) {
      const nx = gx[c]! + DX[d];
      const ny = gy[c]! + DY[d];
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const l = localOf[ny * lv.w + nx];
      if (l < 0) continue;
      nbr[c * 4 + d] = l;
      dest[c * 4 + d] = l;
    }
  }
  const goalList = lv.goals.map((g) => localOf[g]!);
  const isGoal = new Array<boolean>(n).fill(false);
  for (const g of goalList) isGoal[g] = true;
  const boxes0 = lv.boxes.map((b) => localOf[b]!).sort((a, b) => a - b);
  const space: Space = {
    w: lv.w,
    h: lv.h,
    n,
    gx,
    gy,
    localOf,
    walkable,
    nbr,
    dest,
    goalList,
    isGoal,
    dist: [],
    live: [],
    boxes0,
    player0: localOf[lv.player]!,
    entryLocal: -1,
    keyFn: (player, boxes) => `${player}|${boxes.join(',')}`,
  };
  return finishSpace(space);
}

// ---------- heuristic: minimum-cost box-to-goal matching ----------

function heuristic(space: Space, boxes: number[]): number {
  const n = boxes.length;
  const m = space.goalList.length;
  if (n === 0) return 0;
  if (n > 6) {
    // Too many boxes for exact matching: sum of each box's nearest goal.
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let j = 0; j < m; j++) best = Math.min(best, space.dist[b]![j]!);
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  let best = Infinity;
  const used = new Array<boolean>(m).fill(false);
  const assign = (i: number, cost: number): void => {
    if (cost >= best) return;
    if (i === n) {
      best = cost;
      return;
    }
    for (let j = 0; j < m; j++) {
      if (used[j]) continue;
      const d = space.dist[boxes[i]!]![j]!;
      if (d === Infinity) continue;
      used[j] = true;
      assign(i + 1, cost + d);
      used[j] = false;
    }
  };
  assign(0, 0);
  return best;
}

// ---------- deadlock detection ----------

function isDead(space: Space, boxes: number[]): boolean {
  const boxAt = new Uint8Array(space.n);
  for (const b of boxes) boxAt[b] = 1;
  for (const b of boxes) {
    if (space.isGoal[b]) continue;
    if (!space.live[b]) return true;
    // Corner: two perpendicular neighbours the player cannot stand on.
    const up = space.nbr[b * 4 + 0] < 0;
    const dn = space.nbr[b * 4 + 1] < 0;
    const lf = space.nbr[b * 4 + 2] < 0;
    const rt = space.nbr[b * 4 + 3] < 0;
    if ((up && lf) || (up && rt) || (dn && lf) || (dn && rt)) return true;
    // Frozen: no direction can be pushed at all (walls only; a neighbouring
    // box might move away, so boxes never count as walls here).
    let movable = false;
    for (let d = 0; d < 4; d++) {
      if (space.dest[b * 4 + d] >= 0 && space.nbr[b * 4 + OPP[d]] >= 0) {
        movable = true;
        break;
      }
    }
    if (!movable) return true;
    // Filled 2x2 block: no box inside it can move out.
    const bx = space.gx[b]!;
    const by = space.gy[b]!;
    for (let ty = by - 1; ty <= by; ty++) {
      for (let tx = bx - 1; tx <= bx; tx++) {
        let filled = 0;
        let hasBox = false;
        let hasOffGoalBox = false;
        for (let yy = ty; yy <= ty + 1; yy++) {
          for (let xx = tx; xx <= tx + 1; xx++) {
            if (xx < 0 || xx >= space.w || yy < 0 || yy >= space.h) {
              filled++;
              continue;
            }
            const g = space.localOf[yy * space.w + xx];
            if (g < 0 || !space.walkable[g]) {
              filled++;
              continue;
            }
            if (boxAt[g]) {
              filled++;
              hasBox = true;
              if (!space.isGoal[g]) hasOffGoalBox = true;
            }
          }
        }
        if (filled === 4 && hasBox && hasOffGoalBox) return true;
      }
    }
  }
  return false;
}

// ---------- walks (plain steps, no pushes) ----------

// Shortest walk inside a search space, avoiding boxes. Letters are steps.
function walkLocal(space: Space, from: number, to: number, boxes: number[]): string | null {
  if (from === to) return '';
  const boxSet = new Set(boxes);
  const prev = new Int32Array(space.n).fill(-2);
  const prevDir = new Int8Array(space.n).fill(-1);
  const q = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (let d = 0; d < 4; d++) {
      const nb = space.nbr[c * 4 + d];
      if (nb < 0 || prev[nb] !== -2 || boxSet.has(nb)) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        let s = '';
        for (let t = nb; t !== from; t = prev[t]!) s = LETTER[prevDir[t]!] + s;
        return s;
      }
      q.push(nb);
    }
  }
  return null;
}

// Shortest walk across the whole level, avoiding walls and boxes.
function walkGlobal(lv: Level, from: number, to: number, boxes: number[]): string | null {
  if (from === to) return '';
  const boxSet = new Set(boxes);
  const prev = new Int32Array(lv.w * lv.h).fill(-2);
  const prevDir = new Int8Array(lv.w * lv.h).fill(-1);
  const q = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % lv.w;
    const cy = Math.floor(c / lv.w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const n = ny * lv.w + nx;
      if (lv.wall[n] || boxSet.has(n) || prev[n] !== -2) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let s = '';
        for (let t = n; t !== from; t = prev[t]!) s = LETTER[prevDir[t]!] + s;
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// ---------- push A* ----------

interface Push {
  box: number;
  dir: number;
}

function heapPush(heap: [number, number, number, number][], e: [number, number, number, number]): void {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    const a = heap[i]!;
    const b = heap[p]!;
    if (a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])))) {
      heap[i] = b;
      heap[p] = a;
      i = p;
    } else break;
  }
}

function heapPop(heap: [number, number, number, number][]): [number, number, number, number] {
  const top = heap[0]!;
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length) {
        const a = heap[c]!;
        const b = heap[c + 1]!;
        if (b[0] < a[0] || (b[0] === a[0] && (b[1] < a[1] || (b[1] === a[1] && b[2] < a[2])))) c++;
      }
      const cur = heap[i]!;
      const ch = heap[c]!;
      if (ch[0] < cur[0] || (ch[0] === cur[0] && (ch[1] < cur[1] || (ch[1] === cur[1] && ch[2] < cur[2])))) {
        heap[i] = ch;
        heap[c] = cur;
        i = c;
      } else break;
    }
  }
  return top;
}

function pushAStar(
  space: Space,
  cap: number,
  exitTest: (player: number, boxes: number[]) => boolean,
): { pushes: Push[]; expansions: number } | null {
  const keyFn = space.keyFn;
  const players: number[] = [space.player0];
  const boxesArr: number[][] = [space.boxes0.slice()];
  const gArr: number[] = [0];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<number | string, number>();
  best.set(keyFn(space.player0, space.boxes0), 0);
  const heap: [number, number, number, number][] = [];
  let tie = 0;
  // The estimate depends only on the box configuration (not the player
  // position), so memoize it: many states share a box layout with the
  // player standing in different spots.
  const hCache = new Map<string, number>();
  const heuristicCached = (boxes: number[]): number => {
    const k = boxes.join(',');
    const v = hCache.get(k);
    if (v !== undefined) return v;
    const h = heuristic(space, boxes);
    hCache.set(k, h);
    return h;
  };
  const h0 = heuristicCached(space.boxes0);
  if (h0 === Infinity) return null;
  heapPush(heap, [h0, h0, tie++, 0]);
  let expansions = 0;
  while (heap.length > 0) {
    const entry = heapPop(heap);
    const s = entry[3];
    const k = keyFn(players[s]!, boxesArr[s]!);
    if (best.get(k) !== s) continue; // a better path to this state exists
    const player = players[s]!;
    const boxes = boxesArr[s]!;
    if (boxes.every((b) => space.isGoal[b]) && exitTest(player, boxes)) {
      const pushes: Push[] = [];
      for (let t = s; t > 0; t = parent[t]!) pushes.push({ box: pushBox[t]!, dir: pushDir[t]! });
      pushes.reverse();
      return { pushes, expansions };
    }
    if (expansions >= cap || globalExpansions >= GLOBAL_CAP) return null;
    expansions++;
    globalExpansions++;
    const boxAt = new Uint8Array(space.n);
    for (const b of boxes) boxAt[b] = 1;
    const reach = new Uint8Array(space.n);
    const rq = [player];
    reach[player] = 1;
    for (let qi = 0; qi < rq.length; qi++) {
      const c = rq[qi]!;
      for (let d = 0; d < 4; d++) {
        const nb = space.nbr[c * 4 + d];
        if (nb < 0 || reach[nb] || boxAt[nb]) continue;
        reach[nb] = 1;
        rq.push(nb);
      }
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let d = 0; d < 4; d++) {
        const dest = space.dest[b * 4 + d];
        if (dest < 0 || boxAt[dest]) continue;
        const pushPos = space.nbr[b * 4 + OPP[d]];
        if (pushPos < 0 || !reach[pushPos]) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((x, y) => x - y);
        if (isDead(space, nb)) continue;
        const h = heuristicCached(nb);
        if (h === Infinity) continue;
        const g = gArr[s]! + 1;
        const nk = keyFn(b, nb);
        const seen = best.get(nk);
        if (seen !== undefined && gArr[seen]! <= g) continue;
        const t = players.length;
        players.push(b);
        boxesArr.push(nb);
        gArr.push(g);
        parent.push(s);
        pushBox.push(b);
        pushDir.push(d);
        best.set(nk, t);
        heapPush(heap, [g + h, h, tie++, t]);
      }
    }
  }
  return null;
}

// Turn a push sequence into a move string: walk to each push, push, and at
// the end walk back out to the entry cell.
function reconstruct(space: Space, pushes: Push[], boxes0: number[], player0: number): string | null {
  let moves = '';
  let player = player0;
  const boxes = boxes0.slice();
  for (const p of pushes) {
    const pushPos = space.nbr[p.box * 4 + OPP[p.dir]];
    const walk = walkLocal(space, player, pushPos, boxes);
    if (walk === null) return null;
    moves += walk;
    moves += LETTER[p.dir].toUpperCase();
    const dest = space.dest[p.box * 4 + p.dir];
    const bi = boxes.indexOf(p.box);
    if (bi < 0 || dest < 0) return null;
    boxes[bi] = dest;
    boxes.sort((a, b) => a - b);
    player = p.box;
  }
  if (space.entryLocal >= 0) {
    const exit = walkLocal(space, player, space.entryLocal, boxes);
    if (exit === null) return null;
    moves += exit;
  }
  return moves;
}

// ---------- room solver ----------

function solveRoom(lv: Level, room: Room): string | null {
  const space = buildRoomSpace(lv, room);
  const exitTest = (player: number, boxes: number[]): boolean => {
    const boxAt = new Uint8Array(space.n);
    for (const b of boxes) boxAt[b] = 1;
    const seen = new Uint8Array(space.n);
    const q = [player];
    seen[player] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      if (c === space.entryLocal) return true;
      for (let d = 0; d < 4; d++) {
        const nb = space.nbr[c * 4 + d];
        if (nb < 0 || seen[nb] || boxAt[nb]) continue;
        seen[nb] = 1;
        q.push(nb);
      }
    }
    return false;
  };
  const res = pushAStar(space, ROOM_CAP, exitTest);
  if (!res) return null;
  return reconstruct(space, res.pushes, space.boxes0, space.player0);
}

function solveWhole(lv: Level): string | null {
  const space = buildWholeSpace(lv);
  const res = pushAStar(space, FALLBACK_CAP, () => true);
  if (!res) return null;
  return reconstruct(space, res.pushes, space.boxes0, space.player0);
}

// ---------- level solver ----------

function solveLevel(lv: Level): string {
  const rooms = extractRooms(lv);
  if (rooms && rooms.length > 0) {
    rooms.sort(
      (a, b) => (a.door % lv.w) - (b.door % lv.w) || Math.floor(a.door / lv.w) - Math.floor(b.door / lv.w),
    );
    let moves = '';
    let curPlayer = lv.player;
    let curBoxes = lv.boxes.slice();
    for (const room of rooms) {
      const walk = walkGlobal(lv, curPlayer, room.entry, curBoxes);
      if (walk === null) return solveWhole(lv) ?? '';
      moves += walk;
      curPlayer = room.entry;
      const rm = solveRoom(lv, room);
      if (rm === null) return solveWhole(lv) ?? '';
      moves += rm;
      curPlayer = room.entry;
      const roomBoxSet = new Set(room.boxes);
      curBoxes = curBoxes.filter((b) => !roomBoxSet.has(b));
      curBoxes.push(...room.goals);
      curBoxes.sort((a, b) => a - b);
    }
    return moves;
  }
  return solveWhole(lv) ?? '';
}

// ---------- main ----------

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
    moves = solveLevel(parseLevel(text));
  } catch {
    moves = '';
  }
  const dir = join(O, id);
  mkdirSync(dir, { recursive: true });
  for (const name of readdirSync(dir)) rmSync(join(dir, name), { recursive: true, force: true });
  writeFileSync(join(dir, 'solution.txt'), `${moves}\n`);
}
