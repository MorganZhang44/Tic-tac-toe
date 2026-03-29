import time
import argparse
import sys
import os
import pygame
import numpy as np

from models.common.env import Gomoku9x9
from models.cnn_standard.agent import GomokuDQNAgent
from models.cnn_standard.network import DuelingGomokuNet

# ── Colour palette ─────────────────────────────────────────────────────
BG          = (26, 26, 46)     # #1a1a2e dark navy background
PANEL_BG    = (22, 33, 62)     # #16213e slightly lighter panel
CELL_EMPTY  = (15, 52, 96)     # #0f3460 cell fill
CELL_HOVER  = (20, 70, 120)    # hover state
CELL_X      = (233, 69, 96)    # #e94560 Player 1 (X)
CELL_O      = (242, 169, 0)    # #f2a900 Player -1 (O)
WIN_FLASH   = (50, 205, 50)    # lime green for winning pieces 
TEXT_LIGHT  = (230, 230, 250)
TEXT_DIM    = (122, 122, 154)

# ── Metrics ────────────────────────────────────────────────────────────
ROWS = 9
COLS = 9
SQUARE_SIZE = 80
LINE_WIDTH = 2
PAD = 20
INFO_H = 100
BOTTOM_H = 80
W = COLS * SQUARE_SIZE + 2 * PAD
H = INFO_H + ROWS * SQUARE_SIZE + 2 * PAD + BOTTOM_H

# Colors for drawing
BG_COLOR = BG
GRID_COLOR = TEXT_DIM
PLAYER1_COLOR = (0, 0, 0) # Black stone
PLAYER2_COLOR = (255, 255, 255) # White stone


class GameUI:
    def __init__(self, agent: GomokuDQNAgent, human_player: int, ai_vs_ai: bool):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("9×9 Gomoku | Human vs AI")

        self.agent = agent
        self.human_player = human_player
        self.ai_vs_ai = ai_vs_ai

        self.env = Gomoku9x9()
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
        title = self.font_title.render("9×9 GOMOKU", True, CELL_X)
        self.screen.blit(title, (W//2 - title.get_width()//2, 10))

        status = self.font_status.render(self.status_msg, True, TEXT_LIGHT)
        self.screen.blit(status, (W//2 - status.get_width()//2, 45))

        score_text = f"🧑 Human: {self.scores['Human']}   |   Draw: {self.scores['Draw']}   |   🤖 AI: {self.scores['AI']}"
        score_surf = self.font_score.render(score_text, True, TEXT_DIM)
        self.screen.blit(score_surf, (W//2 - score_surf.get_width()//2, 75))

        # ── Board ─────────────────────────────────────────────────────
        mx, my = mouse_pos
        
        # Draw grid lines
        for i in range(ROWS + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (PAD, INFO_H + PAD + i * SQUARE_SIZE), (PAD + COLS * SQUARE_SIZE, INFO_H + PAD + i * SQUARE_SIZE), LINE_WIDTH)
        for i in range(COLS + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (PAD + i * SQUARE_SIZE, INFO_H + PAD), (PAD + i * SQUARE_SIZE, INFO_H + PAD + ROWS * SQUARE_SIZE), LINE_WIDTH)

        for r in range(ROWS):
            for c in range(COLS):
                x0 = PAD + c * SQUARE_SIZE
                y0 = INFO_H + PAD + r * SQUARE_SIZE
                rect = pygame.Rect(x0, y0, SQUARE_SIZE, SQUARE_SIZE)

                is_win = (r, c) in self.winning_cells
                val = self.env.board[r, c]
                center = (x0 + SQUARE_SIZE // 2, y0 + SQUARE_SIZE // 2)

                if is_win and self.flash_state:
                    pygame.draw.circle(self.screen, WIN_FLASH, center, SQUARE_SIZE // 2 - 2)

                # Need a hover effect for empty grids
                if val == 0 and not self.game_over and rect.collidepoint(mx, my):
                    pygame.draw.circle(self.screen, CELL_HOVER, center, SQUARE_SIZE // 2 - 8)

                # Draw pieces
                radius = SQUARE_SIZE // 2 - 6
                if val == 1:
                    pygame.draw.circle(self.screen, PLAYER1_COLOR, center, radius)
                elif val == -1:
                    pygame.draw.circle(self.screen, PLAYER2_COLOR, center, radius)
                    pygame.draw.circle(self.screen, (0,0,0), center, radius, 2) # outline for white

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

        for r in range(ROWS):
            for c in range(COLS):
                x0 = PAD + c * SQUARE_SIZE
                y0 = INFO_H + PAD + r * SQUARE_SIZE
                if pygame.Rect(x0, y0, SQUARE_SIZE, SQUARE_SIZE).collidepoint(mx, my):
                    action = r * COLS + c
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
        if self.winner in (0, None):
            return set()
        
        b = self.env.board
        w = 5  # Gomoku is 5-in-a-row
        cells = set()
        
        # Horiz
        for r in range(ROWS):
            for c in range(COLS - w + 1):
                if all(b[r, c+k] == self.winner for k in range(w)):
                    for k in range(w): cells.add((r, c+k))
        # Vert
        for r in range(ROWS - w + 1):
            for c in range(COLS):
                if all(b[r+k, c] == self.winner for k in range(w)):
                    for k in range(w): cells.add((r+k, c))
        # Diag ↘
        for r in range(ROWS - w + 1):
            for c in range(COLS - w + 1):
                if all(b[r+k, c+k] == self.winner for k in range(w)):
                    for k in range(w): cells.add((r+k, c+k))
        # Diag ↙
        for r in range(ROWS - w + 1):
            for c in range(w - 1, COLS):
                if all(b[r+k, c-k] == self.winner for k in range(w)):
                    for k in range(w): cells.add((r+k, c-k))
                    
        return cells

    def run(self):
        clock = pygame.time.Clock()
        while True:
            # Animation timer
            now = time.time()
            if now - self.last_update > 0.5:
                self.flash_state = not self.flash_state
                self.last_update = now

            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(mouse_pos)

            self.draw(mouse_pos)

            # AI move logic
            if not self.game_over:
                if self.ai_vs_ai or self.env.current_player != self.human_player:
                    pygame.display.flip()  # Ensure status message updates before freeze
                    pygame.time.delay(500) # Give human a chance to see the move
                    self.ai_move()

            clock.tick(60)


def main():
    parser = argparse.ArgumentParser("Play COMP0215 Gomoku AI")
    parser.add_argument("--ai_first", action="store_true", help="AI plays as X (first)")
    parser.add_argument("--ai_vs_ai", action="store_true", help="Watch two AI instances battle")
    args = parser.parse_args()

    weights_path = os.path.join(os.path.dirname(__file__), "models", "cnn_standard", "weights.pth")
    if not os.path.exists(weights_path):
        print(f"Error: No weights found at {weights_path}")
        print("Please train the Gomoku agent first!")
        sys.exit(1)

    agent = GomokuDQNAgent(board_size=9, action_size=81)
    agent.load(weights_path)
    agent.epsilon = 0.0  # Greedy strictly

    human_player = -1 if args.ai_first else 1
    ui = GameUI(agent, human_player=human_player, ai_vs_ai=args.ai_vs_ai)
    ui.run()

if __name__ == "__main__":
    main()
