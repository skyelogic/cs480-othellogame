# -*- coding: utf-8 -*-
"""
Filename: HW4_AdversarialSearch.py
Author: Donnel Garner
Date: 2/12/2026
Description: CS480 Module 3 - Adversarial Search: Othello (Reversi)

Othello: Human vs AI (Minimax + Alpha-Beta Pruning)

How to run:
  HW4_AdversarialSearch.py

Move input formats:
  d3   (recommended)
  4 3  (row col, 1-based)
  pass (only if you have no legal moves)

Board coordinates:
  Columns: a b c d e f g h
  Rows:    1 2 3 4 5 6 7 8

Notes:
- Human can choose Black (B) or White (W). Black moves first.
- AI uses alpha-beta pruning with a heuristic evaluation for intermediate states.
- You can set search depth and (optional) time limit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import time
import sys

# -----------------------------
# Game constants / helpers
# -----------------------------

EMPTY = "."
BLACK = "B"
WHITE = "W"

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE

def in_bounds(r: int, c: int, n: int) -> bool:
    return 0 <= r < n and 0 <= c < n

def parse_move(s: str) -> Optional[Tuple[int, int]]:
    """Parse a move like 'd3' or '4 3'. Returns (row_idx, col_idx) 0-based."""
    s = s.strip().lower()
    if s == "pass":
        return None
    # Format: letter+number e.g., d3
    if len(s) >= 2 and s[0].isalpha():
        col = ord(s[0]) - ord('a')
        num = s[1:]
        if num.isdigit():
            row = int(num) - 1
            return (row, col)
    # Format: "row col"
    parts = s.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        row = int(parts[0]) - 1
        col = int(parts[1]) - 1
        return (row, col)
    return None

def move_to_str(move: Optional[Tuple[int, int]]) -> str:
    if move is None:
        return "pass"
    r, c = move
    return f"{chr(ord('a') + c)}{r+1}"

# -----------------------------
# Othello mechanics
# -----------------------------

@dataclass(frozen=True)
class GameState:
    board: Tuple[Tuple[str, ...], ...]  # immutable for easy hashing
    to_move: str                        # 'B' or 'W'

    @property
    def n(self) -> int:
        return len(self.board)

def initial_state(n: int = 8) -> GameState:
    """Standard 8x8 Othello start position."""
    board = [[EMPTY for _ in range(n)] for _ in range(n)]
    mid = n // 2
    board[mid-1][mid-1] = WHITE
    board[mid][mid]     = WHITE
    board[mid-1][mid]   = BLACK
    board[mid][mid-1]   = BLACK
    return GameState(tuple(tuple(row) for row in board), BLACK)

def print_board(state: GameState) -> None:
    n = state.n
    header = "   " + " ".join(chr(ord('a') + i) for i in range(n))
    print(header)
    for r in range(n):
        row_str = " ".join(state.board[r][c] for c in range(n))
        print(f"{r+1:2d} {row_str}")
    b, w = count_pieces(state)
    print(f"\nScore -> B: {b}   W: {w}   Turn: {state.to_move}")

def count_pieces(state: GameState) -> Tuple[int, int]:
    b = sum(cell == BLACK for row in state.board for cell in row)
    w = sum(cell == WHITE for row in state.board for cell in row)
    return b, w

def find_flips(board: Tuple[Tuple[str, ...], ...], n: int, r: int, c: int, color: str) -> List[Tuple[int, int]]:
    """Return list of opponent disks to flip if placing 'color' at (r,c)."""
    if not in_bounds(r, c, n) or board[r][c] != EMPTY:
        return []
    opp = opponent(color)
    flips: List[Tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        line: List[Tuple[int, int]] = []
        while in_bounds(rr, cc, n) and board[rr][cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and in_bounds(rr, cc, n) and board[rr][cc] == color:
            flips.extend(line)
    return flips

def legal_moves(state: GameState, color: Optional[str] = None) -> List[Tuple[int, int]]:
    """All legal moves for 'color' (defaults to state's to_move)."""
    color = color or state.to_move
    n = state.n
    moves = []
    for r in range(n):
        for c in range(n):
            if state.board[r][c] == EMPTY:
                if find_flips(state.board, n, r, c, color):
                    moves.append((r, c))
    return moves

def apply_move(state: GameState, move: Optional[Tuple[int, int]]) -> GameState:
    """Apply move for state's to_move. If move is None, it's a pass."""
    n = state.n
    color = state.to_move
    if move is None:
        return GameState(state.board, opponent(color))

    r, c = move
    flips = find_flips(state.board, n, r, c, color)
    if not flips:
        raise ValueError("Illegal move attempted.")

    # Create mutable copy
    new_board = [list(row) for row in state.board]
    new_board[r][c] = color
    for rr, cc in flips:
        new_board[rr][cc] = color

    return GameState(tuple(tuple(row) for row in new_board), opponent(color))

def game_over(state: GameState) -> bool:
    """Game ends when neither player has a legal move."""
    return (len(legal_moves(state, BLACK)) == 0) and (len(legal_moves(state, WHITE)) == 0)

# -----------------------------
# Heuristic evaluation
# -----------------------------

# Position weights: corners are huge, edges are good, squares adjacent to corners are risky early.
# (Common Othello heuristic pattern; works well for intermediate play.)
WEIGHTS_8x8 = (
    (120, -20,  20,   5,   5,  20, -20, 120),
    (-20, -40,  -5,  -5,  -5,  -5, -40, -20),
    ( 20,  -5,  15,   3,   3,  15,  -5,  20),
    (  5,  -5,   3,   3,   3,   3,  -5,   5),
    (  5,  -5,   3,   3,   3,   3,  -5,   5),
    ( 20,  -5,  15,   3,   3,  15,  -5,  20),
    (-20, -40,  -5,  -5,  -5,  -5, -40, -20),
    (120, -20,  20,   5,   5,  20, -20, 120),
)

def frontier_count(state: GameState, color: str) -> int:
    """Frontier disks are adjacent to empty squares."""
    n = state.n
    board = state.board
    cnt = 0
    for r in range(n):
        for c in range(n):
            if board[r][c] != color:
                continue
            # adjacent to empty?
            for dr, dc in DIRECTIONS:
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc, n) and board[rr][cc] == EMPTY:
                    cnt += 1
                    break
    return cnt

def positional_score(state: GameState, color: str) -> int:
    """Weighted square score (only defined nicely for 8x8 here)."""
    if state.n != 8:
        # For non-8 sizes, fall back to simple edge/corner preference.
        n = state.n
        score = 0
        board = state.board
        corners = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
        for r in range(n):
            for c in range(n):
                if board[r][c] == color:
                    # corners worth a lot
                    if (r,c) in corners:
                        score += 50
                    # edges worth some
                    elif r in (0, n-1) or c in (0, n-1):
                        score += 10
                    else:
                        score += 1
                elif board[r][c] == opponent(color):
                    if (r,c) in corners:
                        score -= 50
                    elif r in (0, n-1) or c in (0, n-1):
                        score -= 10
                    else:
                        score -= 1
        return score

    score = 0
    for r in range(8):
        for c in range(8):
            if state.board[r][c] == color:
                score += WEIGHTS_8x8[r][c]
            elif state.board[r][c] == opponent(color):
                score -= WEIGHTS_8x8[r][c]
    return score

def evaluate(state: GameState, ai_color: str) -> float:
    """
    Heuristic: combine
      - piece difference (small weight early)
      - mobility difference (legal moves)
      - frontier disks (minimize ours, maximize theirs)
      - positional weights (corners/edges)
    """
    b, w = count_pieces(state)
    my_pieces = b if ai_color == BLACK else w
    opp_pieces = w if ai_color == BLACK else b
    piece_diff = my_pieces - opp_pieces

    my_moves = len(legal_moves(state, ai_color))
    opp_moves = len(legal_moves(state, opponent(ai_color)))
    mobility = my_moves - opp_moves

    my_front = frontier_count(state, ai_color)
    opp_front = frontier_count(state, opponent(ai_color))
    frontier = opp_front - my_front  # good if opponent has more frontier disks

    pos = positional_score(state, ai_color)

    # Game phase scaling: early vs late (based on filled squares)
    n = state.n
    filled = (n*n) - sum(cell == EMPTY for row in state.board for cell in row)
    phase = filled / (n*n)

    # Weights (tuned for decent play while staying simple/explainable)
    # Early: prioritize mobility/position; Late: piece count matters more.
    w_piece = 2 + 20*phase
    w_mob   = 25 - 10*phase
    w_front = 10 -  5*phase
    w_pos   = 30

    return (w_piece * piece_diff) + (w_mob * mobility) + (w_front * frontier) + (w_pos * (pos / 100.0))

# -----------------------------
# Minimax + Alpha-Beta
# -----------------------------

@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    start_time: float = 0.0
    time_limit: Optional[float] = None  # seconds
    aborted: bool = False

def time_exceeded(stats: SearchStats) -> bool:
    if stats.time_limit is None:
        return False
    return (time.time() - stats.start_time) >= stats.time_limit

def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_color: str,
    ai_color: str,
    stats: SearchStats,
) -> float:
    """
    Returns heuristic value from AI's perspective.
    maximizing_color: whose turn is maximizing at this node (ai_color when AI to move).
    """
    stats.nodes += 1
    if time_exceeded(stats):
        stats.aborted = True
        return evaluate(state, ai_color)

    if depth == 0 or game_over(state):
        # If game over, return a strong terminal score based on final outcome.
        if game_over(state):
            b, w = count_pieces(state)
            diff = (b - w) if ai_color == BLACK else (w - b)
            # Huge terminal magnitude so mate/win dominates.
            return 10_000 + diff if diff > 0 else -10_000 + diff if diff < 0 else 0
        return evaluate(state, ai_color)

    moves = legal_moves(state, state.to_move)
    # If no legal moves, must pass (do not reduce depth too aggressively, but do progress)
    if not moves:
        return alphabeta(apply_move(state, None), depth - 1, alpha, beta, opponent(maximizing_color), ai_color, stats)

    if state.to_move == maximizing_color:
        value = -math.inf
        for mv in moves:
            child = apply_move(state, mv)
            value = max(value, alphabeta(child, depth - 1, alpha, beta, maximizing_color, ai_color, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value
    else:
        value = math.inf
        for mv in moves:
            child = apply_move(state, mv)
            value = min(value, alphabeta(child, depth - 1, alpha, beta, maximizing_color, ai_color, stats))
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value

def choose_ai_move(state: GameState, ai_color: str, depth: int, time_limit: Optional[float]) -> Tuple[Optional[Tuple[int, int]], SearchStats]:
    """
    Choose best move for AI using alpha-beta.
    Uses a simple move ordering: prefer corners, then high positional weight.
    """
    stats = SearchStats(start_time=time.time(), time_limit=time_limit)
    moves = legal_moves(state, ai_color)
    if not moves:
        return None, stats

    # Order moves: corners first, then positional weight
    def move_key(mv: Tuple[int, int]) -> float:
        r, c = mv
        if state.n == 8:
            return WEIGHTS_8x8[r][c]
        # fallback: corners/edges
        n = state.n
        if (r, c) in [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]:
            return 999
        if r in (0,n-1) or c in (0,n-1):
            return 50
        return 0

    ordered = sorted(moves, key=move_key, reverse=True)

    best_move = ordered[0]
    best_val = -math.inf
    alpha, beta = -math.inf, math.inf

    # Root search (maximizing for AI)
    for mv in ordered:
        child = apply_move(state, mv)
        val = alphabeta(child, depth - 1, alpha, beta, maximizing_color=ai_color, ai_color=ai_color, stats=stats)
        if stats.aborted:
            # Time limit hit; return best found so far.
            break
        if val > best_val:
            best_val = val
            best_move = mv
        alpha = max(alpha, best_val)

    return best_move, stats

# -----------------------------
# Main game loop
# -----------------------------

def ask_choice(prompt: str, valid: Dict[str, str]) -> str:
    """valid maps user_input -> internal value."""
    while True:
        s = input(prompt).strip().lower()
        if s in valid:
            return valid[s]
        print(f"Please choose one of: {', '.join(valid.keys())}")

def main() -> None:
    print("=== Othello (Reversi) - Human vs AI (Minimax + Alpha-Beta) ===\n")

    # Player chooses color
    human_color = ask_choice("Choose your color ([b]lack goes first / [w]hite): ", {"b": BLACK, "black": BLACK, "w": WHITE, "white": WHITE})
    ai_color = opponent(human_color)

    # Difficulty settings
    try:
        depth = int(input("Choose AI search depth (recommended 3-5): ").strip() or "4")
    except ValueError:
        depth = 4
    if depth < 1:
        depth = 1

    # Optional time limit to prevent "running forever"
    t_raw = input("Optional AI time limit per move in seconds (blank = no limit, recommended 2-5): ").strip()
    time_limit = None
    if t_raw:
        try:
            time_limit = float(t_raw)
            if time_limit <= 0:
                time_limit = None
        except ValueError:
            time_limit = None

    state = initial_state(8)
    consecutive_passes = 0

    while True:
        print("\n" + "-"*40)
        print_board(state)

        if game_over(state):
            b, w = count_pieces(state)
            print("\nGame Over!")
            if b > w:
                print("Winner: BLACK")
            elif w > b:
                print("Winner: WHITE")
            else:
                print("Result: DRAW")
            break

        current = state.to_move
        moves = legal_moves(state, current)

        if not moves:
            print(f"{current} has no legal moves and must PASS.")
            state = apply_move(state, None)
            consecutive_passes += 1
            if consecutive_passes >= 2:
                # should also be game over, but keep safe
                continue
            continue
        else:
            consecutive_passes = 0

        if current == human_color:
            # Human turn
            print(f"\nYour turn ({human_color}). Legal moves: {', '.join(move_to_str(m) for m in moves)}")
            while True:
                raw = input("Enter your move (e.g., d3) or 'pass': ").strip()
                mv = parse_move(raw)
                if mv is None:
                    # Allow pass only if truly no moves (but we already have moves)
                    print("You have legal moves — you cannot pass right now.")
                    continue
                if mv in moves:
                    state = apply_move(state, mv)
                    break
                else:
                    print("Illegal move. Try again.")
        else:
            # AI turn
            print(f"\nAI thinking... ({ai_color}) depth={depth}" + (f", time_limit={time_limit}s" if time_limit else ""))
            mv, stats = choose_ai_move(state, ai_color, depth, time_limit)
            if mv is None:
                print("AI has no legal moves and passes.")
                state = apply_move(state, None)
            else:
                print(f"AI plays: {move_to_str(mv)}  | nodes={stats.nodes}, cutoffs={stats.cutoffs}" + (" (TIME LIMIT HIT)" if stats.aborted else ""))
                state = apply_move(state, mv)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting game.")
        sys.exit(0)
