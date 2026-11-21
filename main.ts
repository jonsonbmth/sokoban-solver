// Sokoban solver: push-optimal A* per room. Rooms are independent (a box pushed
// into the hallway can never come back), so each room's 3-4 boxes are solved
// separately and the move strings concatenated. Deterministic: every budget is
// a node count, never the clock; no randomness, no environment reads.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [[0, -1, 'u', 'U'], [0, 1, 'd', 'D'], [-1, 0, 'l', 'L'], [1, 0, 'r', 'R']];
// Node budgets (work, never time). Rooms are guaranteed solvable by plain BFS
// within 250k states, so A* with a good heuristic should need far less.
const ASTAR_CAP = 300_000;
const BFS_CAP = 1_000_000;
const GREEDY_CAP = 200_000;
const GLOBAL_CAP = 2_000_000;
let spent = 0;

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box can reach some goal, by pulling back from the goals.
function liveSquares(w, wall, goals) {
  const live = new Uint8Array(wall.length);
  const q = [];
  for (const g of goals) { live[g] = 1; q.push(g); }
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
    }
  }
  return live;
}

// Flood the player's reachable region from `from` over open, unoccupied squares.
// Marks it in `seen` with `stamp`, returns the region's lowest square.
function flood(w, wall, occ, from, seen, stamp, stack) {
  let low = from, n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    }
  }
  return low;
}

// BFS distance from every square to each goal (walls only, no boxes).
function goalDistances(w, wall, goals) {
  const dists = [];
  for (const g of goals) {
    const d = new Int32Array(wall.length).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let i = 0; i < q.length; i++) {
      const p = q[i];
      for (const [dx, dy] of DIRS) {
        const n = p + dy * w + dx;
        if (!wall[n] && d[n] < 0) { d[n] = d[p] + 1; q.push(n); }
      }
    }
    dists.push(d);
  }
  return dists;
}

// Min-cost box->goal assignment over the precomputed distances (admissible:
// each push moves a box at most one step closer to its goal).
function assignmentH(boxes, dists) {
  const n = boxes.length;
  const order = dists.map((_, i) => i);
  let best = Infinity;
  const permute = (k, acc) => {
    if (acc >= best) return;
    if (k === n) { best = acc; return; }
    for (let i = k; i < n; i++) {
      [order[k], order[i]] = [order[i], order[k]];
      const d = dists[order[k]][boxes[k]];
      if (d >= 0) permute(k + 1, acc + d);
      [order[k], order[i]] = [order[i], order[k]];
    }
  };
  permute(0, 0);
  return best;
}

// A frozen 2x2 block (walls and boxes) holding an off-goal box at/near `t`.
function frozen2x2(w, wall, occ, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (lowercase LURD).
function walkTo(w, wall, occ, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const n = p + dy * w + dx;
      if (wall[n] || occ[n] || prev[n] !== -1) continue;
      prev[n] = p;
      how[n] = letter;
      q.push(n);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// Push search over one board: mode 'astar' (optimal), 'bfs' (optimal, no
// heuristic) or 'greedy' (fast, suboptimal fallback). Returns {pushes, moves}
// or null. `size` is w*h (numeric keys need size**(boxes+1) < 2**53).
function pushSearch(board, mode, cap) {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = board;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  if (startBoxes.some((b) => !live[b])) return null;
  const dists = mode === 'bfs' ? null : goalDistances(w, wall, goals);
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (bs) => bs.every((b) => goal[b]);
  const keyOf = (bs, low) => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const hOf = (bs) => (mode === 'bfs' ? 0 : assignmentH(bs, dists));

  const sBoxes = [startBoxes];
  const sPlayer = [startPlayer];
  const parent = [-1];
  const pushFrom = [0];
  const pushTo = [0];
  const gScore = [0];
  const seenAt = new Map();
  for (const b of startBoxes) occ[b] = 1;
  seenAt.set(keyOf(startBoxes, flood(w, wall, occ, startPlayer, seen, ++stamp, stack)), 0);
  for (const b of startBoxes) occ[b] = 0;
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // Binary heap; order depends on mode. seq breaks ties deterministically.
  const heap = [];
  let seq = 0;
  const rank = (g, h) => mode === 'greedy' ? h : mode === 'bfs' ? g : g + h;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  hPush([rank(0, hOf(startBoxes)), 0, seq++]);
  let expanded = 0;

  while (heap.length > 0) {
    const [, g, s] = hPop();
    if (gScore[s] !== g) continue;
    const boxes = sBoxes[s];
    if (solved(boxes)) {
      // Rebuild the move string: walk to each push, then push.
      const path = [];
      for (let t = s; t > 0; t = parent[t]) path.push(t);
      path.reverse();
      let moves = '';
      let player = startPlayer;
      let cur = startBoxes;
      for (const t of path) {
        const from = pushFrom[t], to = pushTo[t];
        const d = to - from;
        const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
        for (const b of cur) occ[b] = 1;
        const walk = walkTo(w, wall, occ, player, from - d);
        for (const b of cur) occ[b] = 0;
        if (walk === null) return null;
        moves += walk + dir[3];
        player = from;
        cur = sBoxes[t];
      }
      return { pushes: path.length, moves };
    }
    if (++expanded > cap || ++spent > GLOBAL_CAP) return null;
    for (const b of boxes) occ[b] = 1;
    const region = ++stamp;
    flood(w, wall, occ, sPlayer[s], seen, region, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== region || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        let dead = frozen2x2(w, wall, occ, goal, t);
        let next = null, key = -1;
        if (!dead) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const low = flood(w, wall, occ, b, mark, ++stamp, stack);
          key = keyOf(next, low);
        }
        occ[t] = 0;
        occ[b] = 1;
        if (dead) continue;
        const prev = seenAt.get(key);
        if (prev !== undefined && gScore[prev] <= g + 1) continue;
        const h = hOf(next);
        if (h === Infinity) continue;
        seenAt.set(key, sBoxes.length);
        sBoxes.push(next);
        sPlayer.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        gScore.push(g + 1);
        hPush([rank(g + 1, h), g + 1, seq++]);
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  return null;
}

// The level's rooms: cut the dead squares of the player's row (the hallway),
// and each remaining part holding boxes is a room. Null when not a room level.
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveRoom(board) {
  return pushSearch(board, 'astar', ASTAR_CAP)
    ?? pushSearch(board, 'bfs', BFS_CAP)
    ?? pushSearch(board, 'greedy', GREEDY_CAP);
}

function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = pushSearch(level, 'astar', ASTAR_CAP) ?? pushSearch(level, 'greedy', GREEDY_CAP);
    return whole ? whole.moves : '';
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = {
      w: level.w, h: level.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveRoom(sub);
    if (solved === null) return '';
    moves += solved.moves;
    for (const ch of solved.moves.toLowerCase())
      player += ch === 'u' ? -level.w : ch === 'd' ? level.w : ch === 'l' ? -1 : 1;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(INPUT).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
