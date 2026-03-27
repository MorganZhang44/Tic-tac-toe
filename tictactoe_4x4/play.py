"""
Interactive 4×4 Connect-4 — Human vs AI (Pygame UI)
======================================================
Play against the trained DQN agent in a visual interface.

Usage:
    python play.py              # You play as X (first), AI plays as O
    python play.py --ai-first   # AI goes first, you play as O
    python play.py --ai-vs-ai   # Watch the AI play itself

Requirements: pygame
"""

import argparse
import os
import sys
import time

# Hide the pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

from game.tictactoe4x4 import TicTacToe4x4
from agent.dqn_agent import DQNAgent

# ── Colour palette ─────────────────────────────────────────────────────
BG          = (26, 26, 46)     # #1a1a2e dark navy background
PANEL_BG    = (22, 33, 62)     # #16213e
CELL_EMPTY  = (15, 52, 96)     # #0f3460 dark blue cell
CELL_X      = (233, 69, 96)    # #e94560 vivid pink-red (human / X)
CELL_O      = (83, 216, 251)   # #53d8fb cyan (AI / O)
CELL_HOVER  = (26, 74, 122)    # #1a4a7a hover
WIN_FLASH   = (245, 166, 35)   # #f5a623 golden flash
TEXT_LIGHT  = (224, 224, 224)
TEXT_DIM    = (122, 122, 154)

# ── Metrics ────────────────────────────────────────────────────────────
BOARD = 4
CELL_SIZE = 120
PAD = 20
INFO_H = 100
BOTTOM_H = 80
W = BOARD * CELL_SIZE + 2 * PAD
H = INFO_H + BOARD * CELL_SIZE + 2 * PAD + BOTTOM_H


class GameUI:
    def __init__(self, agent: DQNAgent, human_player: int, ai_vs_ai: bool):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("4×4 Connect-4 | Human vs AI")

        self.agent = agent
        self.human_player = human_player
        self.ai_vs_ai = ai_vs_ai

        self.env = TicTacToe4x4()
        self.board_state = self.env.reset()
        self.game_over = False
        self.winner = None

        self.scores = {"Human": 0, "AI": 0, "Draw": 0}
        self.winning_cells = set()

        # Fonts
        self.font_title  = pygame.font.SysFont("helvetica", 32, bold=True)
        self.font_status = pygame.font.SysFont("helvetica", 24)
        self.font_score  = pygame.font.SysFont("helvetica", 20)
        self.font_piece  = pygame.font.SysFont("helvetica", 80, bold=True)
        self.font_btn    = pygame.font.SysFont("helvetica", 24, bold=True)

        self.status_msg = "Your turn! Click a cell to play." if not ai_vs_ai else "AI vs AI mode"
        if not ai_vs_ai and self.env.current_player != self.human_player:
            self.status_msg = "🤖 AI is thinking..."

        # Flash animation
        self.flash_timer = 0
        self.flash_state = True
        self.last_update = time.time()

        # Layout tracking
        self.new_game_rect = pygame.Rect(W//2 - 120, H - BOTTOM_H + 20, 110, 40)
        self.quit_rect     = pygame.Rect(W//2 + 10, H - BOTTOM_H + 20, 110, 40)

    def draw(self, mouse_pos):
        self.screen.fill(BG)

        # ── Top Panel ─────────────────────────────────────────────────
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, INFO_H))
        title = self.font_title.render("4×4 CONNECT-4", True, CELL_X)
        self.screen.blit(title, (W//2 - title.get_width()//2, 10))

        status = self.font_status.render(self.status_msg, True, TEXT_LIGHT)
        self.screen.blit(status, (W//2 - status.get_width()//2, 45))

        score_text = f"🧑 Human: {self.scores['Human']}   |   Draw: {self.scores['Draw']}   |   🤖 AI: {self.scores['AI']}"
        score_surf = self.font_score.render(score_text, True, TEXT_DIM)
        self.screen.blit(score_surf, (W//2 - score_surf.get_width()//2, 75))

        # ── Board ─────────────────────────────────────────────────────
        mx, my = mouse_pos
        for r in range(BOARD):
            for c in range(BOARD):
                x0 = PAD + c * CELL_SIZE
                y0 = INFO_H + PAD + r * CELL_SIZE
                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)

                is_win = (r, c) in self.winning_cells

                # Board cell inner rect
                margin = 6
                inner_rect = pygame.Rect(x0 + margin, y0 + margin, 
                                         CELL_SIZE - 2*margin, CELL_SIZE - 2*margin)

                val = self.env.board[r, c]

                if is_win and self.flash_state:
                    color = WIN_FLASH
                elif val == 1:
                    color = CELL_X
                elif val == -1:
                    color = CELL_O
                elif not self.game_over and inner_rect.collidepoint(mx, my):
                    color = CELL_HOVER
                else:
                    color = CELL_EMPTY

                pygame.draw.rect(self.screen, color, inner_rect, border_radius=8)

                # Draw pieces
                if val == 1:
                    piece = self.font_piece.render("X", True, (255, 255, 255))
                    self.screen.blit(piece, (x0 + CELL_SIZE//2 - piece.get_width()//2,
                                             y0 + CELL_SIZE//2 - piece.get_height()//2 + 5))
                elif val == -1:
                    piece = self.font_piece.render("O", True, PANEL_BG)
                    self.screen.blit(piece, (x0 + CELL_SIZE//2 - piece.get_width()//2,
                                             y0 + CELL_SIZE//2 - piece.get_height()//2 + 5))
                elif not self.game_over and inner_rect.collidepoint(mx, my) and not self.ai_vs_ai:
                    if self.env.current_player == self.human_player:
                        sym = "X" if self.human_player == 1 else "O"
                        text_col = (255, 255, 255, 128) if self.human_player == 1 else (*PANEL_BG, 128)
                        piece = self.font_piece.render(sym, True, text_col)
                        piece.set_alpha(100) # Ghost effect
                        self.screen.blit(piece, (x0 + CELL_SIZE//2 - piece.get_width()//2,
                                                 y0 + CELL_SIZE//2 - piece.get_height()//2 + 5))

        # ── Bottom Buttons ────────────────────────────────────────────
        ng_color = CELL_X if self.new_game_rect.collidepoint(mx, my) else (200, 50, 80)
        pygame.draw.rect(self.screen, ng_color, self.new_game_rect, border_radius=6)
        ng_text = self.font_btn.render("New Game", True, (255, 255, 255))
        self.screen.blit(ng_text, (self.new_game_rect.centerx - ng_text.get_width()//2,
                                   self.new_game_rect.centery - ng_text.get_height()//2))

        q_color = PANEL_BG if self.quit_rect.collidepoint(mx, my) else (30, 40, 70)
        pygame.draw.rect(self.screen, q_color, self.quit_rect, border_radius=6)
        q_text = self.font_btn.render("Quit", True, TEXT_LIGHT)
        self.screen.blit(q_text, (self.quit_rect.centerx - q_text.get_width()//2,
                                  self.quit_rect.centery - q_text.get_height()//2))

        pygame.display.flip()

    def handle_click(self, mouse_pos):
        mx, my = mouse_pos

        if self.new_game_rect.collidepoint(mx, my):
            self.reset_game()
            return

        if self.quit_rect.collidepoint(mx, my):
            pygame.quit()
            sys.exit()

        if self.game_over or self.ai_vs_ai:
            return

        if self.env.current_player != self.human_player:
            return

        for r in range(BOARD):
            for c in range(BOARD):
                x0 = PAD + c * CELL_SIZE
                y0 = INFO_H + PAD + r * CELL_SIZE
                if pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE).collidepoint(mx, my):
                    action = r * BOARD + c
                    if action in self.env.get_valid_moves():
                        self.apply_move(action, is_human=True)
                    return

    def apply_move(self, action, is_human):
        _, reward, done, info = self.env.step(action)
        
        if done:
            self.game_over = True
            self.winner = info["winner"]
            self.winning_cells = self.find_winning_cells()
            
            if self.winner == 0:
                self.status_msg = "🤝 It's a draw!"
                self.scores["Draw"] += 1
            elif self.winner == self.human_player:
                self.status_msg = "🎉 You win! Impressive!"
                self.scores["Human"] += 1
            else:
                self.status_msg = "🤖 AI wins! Better luck next time."
                self.scores["AI"] += 1
        else:
            if not is_human and not self.ai_vs_ai:
                self.status_msg = "Your turn! Click a cell to play."
            else:
                self.status_msg = "🤖 AI is thinking..."

    def ai_move(self):
        player = self.env.current_player
        mask = self.env.get_valid_mask()
        state = self.env.get_state_for_player(player)
        action = self.agent.select_action(state, mask, greedy=True)
        self.apply_move(action, is_human=False)

    def reset_game(self):
        self.env.reset()
        self.game_over = False
        self.winner = None
        self.winning_cells = set()
        if self.ai_vs_ai:
            self.status_msg = "🤖 AI vs AI — watching self-play..."
        elif self.env.current_player != self.human_player:
            self.status_msg = "🤖 AI is thinking..."
        else:
            self.status_msg = "Your turn! Click a cell to play."

    def find_winning_cells(self):
        b = self.env.board
        n = BOARD
        w = 4
        for player in [1, -1]:
            for r in range(n):
                for c in range(n - w + 1):
                    if all(b[r, c+k] == player for k in range(w)): return {(r, c+k) for k in range(w)}
            for r in range(n - w + 1):
                for c in range(n):
                    if all(b[r+k, c] == player for k in range(w)): return {(r+k, c) for k in range(w)}
            for r in range(n - w + 1):
                for c in range(n - w + 1):
                    if all(b[r+k, c+k] == player for k in range(w)): return {(r+k, c+k) for k in range(w)}
            for r in range(n - w + 1):
                for c in range(w - 1, n):
                    if all(b[r+k, c-k] == player for k in range(w)): return {(r+k, c-k) for k in range(w)}
        return set()

    def run(self):
        clock = pygame.time.Clock()
        ai_timer = 0

        while True:
            mouse_pos = pygame.mouse.get_pos()
            dt = time.time() - self.last_update
            self.last_update = time.time()

            # Handle flash animation
            if self.game_over and self.winning_cells:
                self.flash_timer += dt
                if self.flash_timer > 0.3:
                    self.flash_state = not self.flash_state
                    self.flash_timer = 0

            # Handle AI moves
            if not self.game_over:
                if (self.ai_vs_ai) or (self.env.current_player != self.human_player):
                    ai_timer += dt
                    if ai_timer > 0.5:  # 500ms delay for visual effect
                        self.ai_move()
                        ai_timer = 0
                else:
                    ai_timer = 0 # reset timer if it's human's turn

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(mouse_pos)

            self.draw(mouse_pos)
            clock.tick(60)

def main():
    parser = argparse.ArgumentParser(description="Play 4×4 Connect-4 against the trained DQN agent")
    parser.add_argument("--ai-first", action="store_true", help="AI goes first (plays as X), you play as O")
    parser.add_argument("--ai-vs-ai", action="store_true", help="Watch the AI play against itself")
    parser.add_argument("--weights", type=str, default=os.path.join("weights", "model_weights.pth"))
    args = parser.parse_args()

    agent = DQNAgent()
    if os.path.exists(args.weights):
        agent.load(args.weights)
        agent.policy_net.eval()
        print(f"[UI] Loaded weights from {args.weights}")
    else:
        print(f"[UI] WARNING: No weights found at '{args.weights}'. Playing randomly.")

    human_player = -1 if args.ai_first else 1
    ui = GameUI(agent, human_player, args.ai_vs_ai)
    ui.run()

if __name__ == "__main__":
    main()
