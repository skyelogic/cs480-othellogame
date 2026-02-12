# CS480 – Adversarial Search - Othello (Reversi)
Author: Donnel Garner  
Date: 02/12/2026  

How to run?  
python HW4_AdversarialSearch.py

This project implements the game Othello (Reversi) as a Human vs AI
console-based Python application using adversarial search.

The AI uses:
- Minimax search
- Alpha-Beta pruning
- A heuristic evaluation function

The program allows the user to:
- Choose to play as Black or White
- Select search depth
- Runs entirely in terminal / command prompt

WEBSITE: https://donnelgarner.com/projects/CS480/othello/index.html  
GOOGLE COLAB: N/A  
GITHUB: https://github.com/skyelogic/cs480-othellogame/  

## GAME RULES

- The board is 8x8.  
- Black moves first.  
- A move is legal if it flips at least one opponent disk.  
- Disks are flipped when bracketed between two disks of the same color.  
- If a player has no legal moves, they must pass.  
- The game ends when neither player has a legal move.  
- The winner is the player with the most disks.  

## AI ALGORITHM
The AI uses Minimax with Alpha-Beta pruning.
- Prunes branches that cannot influence final decision.
- Improves efficiency compared to plain minimax.

## HEURISTIC EVALUATION FUNCTION

1) Piece Difference  
   AI pieces - Opponent pieces  
   Weighted more heavily in late game.  

2) Mobility  
   Number of legal moves available to AI minus opponent.  
   More important in early game.  

3) Positional Weights  
   Uses a predefined 8x8 weight matrix:  
    - Corners highly valuable
    - Squares next to corners penalized
    - Edges positive

4) Phase Scaling  
   As the board fills up:  
    - Piece difference becomes more important
    - Mobility importance decreases

Terminal states return very large positive or negative values  
to prioritize wins over heuristic estimates.

## SEARCH DEPTH
User-selected search depth.  

Recommended:  
- Depth 3 → Fast  
- Depth 4 → Balanced  
- Depth 5 → Stronger AI (slower)  

Higher depths increase node expansions exponentially.  

## REFERENCES
Russell & Norvig, Artificial Intelligence: A Modern Approach  
https://code.claude.com/docs/en/overview  
https://code.visualstudio.com/  
https://github.com/skyelogic/cs480-othellogame/  
https://donnelgarner.com/projects/CS480/othello/index.html  
