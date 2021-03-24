import numpy as np
import pygame
import sys
import math

ROW_COUNT = 6
COLUMN_COUNT = 7

SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

def is_valid_location(board, col):
    return board[0][col] == 0

def move_random(board, turn):
    x = np.array(np.where(board[0] == 0))[0]
    i = np.random.permutation(np.arange(x.size))[0]
    for n in range(ROW_COUNT):
        if board[ROW_COUNT-n-1][x[i]] == 0:
            board[ROW_COUNT-n-1][x[i]] = turn
            return board

def move_manual(board, turn, col):
    for n in range(ROW_COUNT):
        if board[ROW_COUNT-n-1][col] == 0:
            board[ROW_COUNT-n-1][col] = turn
            return board

def move_with_minimax(board, turn):
    best_move = minimax(board, 4, -math.inf, math.inf, turn, turn, True)[1]
    board = move_manual(board.copy(), turn, best_move)
    return board

# board - current node board state
# depth - depth to go for this node
# turn - whos turn overall
# cur_turn - whos turn is this node
# maximizing - is this node a maximize node
def minimax(board, depth, alpha, beta, turn, cur_turn, maximizing):
    # if it's the end node or game has fished, get the heuristic value
    if depth == 0 or have_winner(board, -1) or have_winner(board, 1) or no_available_move(board):
        return get_heuristic_value(board, turn), 0
    # if this is a maximize node, get the max child
    if maximizing:
        v_max = -math.inf
        for c in range(COLUMN_COUNT):
            if is_valid_location(board, c):
                eval= minimax(move_manual(board.copy(), cur_turn, c), depth - 1, alpha, beta, turn, -cur_turn, False)[0]
                if eval > v_max:
                    v_max = eval
                    best_move = c
                alpha = max(alpha, eval)
                if alpha >= beta:
                    break
        return v_max, best_move
    # if this is a minimize node, get the min child
    else:
        v_min = math.inf
        for c in range(COLUMN_COUNT):
            if is_valid_location(board, c):
                eval = minimax(move_manual(board.copy(), cur_turn, c), depth - 1, alpha, beta, turn, -cur_turn, True)[0]
                if eval < v_min:
                    v_min =  eval
                    best_move = c
                beta = min(beta, eval)
                if alpha >= beta:
                    break
        return v_min, best_move

# get the heuristic evaluation of current board state
def get_heuristic_value(board, turn):
    if have_winner(board, turn): return 1000
    if have_winner(board, -turn): return -1000
    eval = 0
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == turn:
                eval += get_piece_heuristic_value(board, r, c, turn)
            if board[r][c] == -turn:
                eval -= get_piece_heuristic_value(board, r, c, -turn)
    return eval

# Check pieces on potential winning position of this piece
def get_piece_heuristic_value(board, r, c, turn):
    p_value = 0
    # (r-3,c-3) (r-2,c-2) (r-1,c-1) (r,c)
    if r > 2 and c > 2:
        if board[r-3][c-3] != -turn and board[r-2][c-2] != -turn and board[r-1][c-1] != -turn:
            p_value += (board[r-3][c-3] == turn) + (board[r-2][c-2] == turn) + (board[r-1][c-1] == turn) + (board[r][c] == turn)
    # (r-2,c-2) (r-1,c-1) (r,c) (r+1,c+1)
    if r > 1 and r < ROW_COUNT-1 and c > 1 and c < COLUMN_COUNT-1:
        if board[r-2][c-2] != -turn and board[r-1][c-1] != -turn and board[r+1][c+1] != -turn:
            p_value += (board[r-2][c-2] == turn) + (board[r-1][c-1] == turn) + (board[r][c] == turn) + (board[r+1][c+1] == turn)
    # (r-1,c-1) (r,c) (r+1,c+1) (r+2,c+2)
    if r > 0 and r < ROW_COUNT-2 and c > 0 and c < COLUMN_COUNT-2:
        if board[r-1][c-1] != -turn and board[r+1][c+1] != -turn and board[r+2][c+2] != -turn :
            p_value += (board[r-1][c-1] == turn) + (board[r][c] == turn) + (board[r+1][c+1] == turn) + (board[r+2][c+2] == turn)
    # (r,c) (r+1,c+1) (r+2,c+2) (r+3,c+3)
    if r < ROW_COUNT-3 and c < COLUMN_COUNT-3:
        if board[r+1][c+1] != -turn and board[r+2][c+2] != -turn and board[r+3][c+3] != -turn :
            p_value += (board[r][c] == turn) + (board[r+1][c+1] == turn) + (board[r+2][c+2] == turn) + (board[r+3][c+3] == turn)
    # (r-3,c+3) (r-2,c+2) (r-1,c+1) (r,c)
    if r > 2 and c < COLUMN_COUNT-3:
        if board[r-3][c+3] != -turn and board[r-2][c+2] != -turn and board[r-1][c+1] != -turn:
            p_value += (board[r-3][c+3] == turn) + (board[r-2][c+2] == turn) + (board[r-1][c+1] == turn) + (board[r][c] == turn)
    # (r-2,c+2) (r-1,c+1) (r,c) (r+1,c-1) 
    if r > 1 and r < ROW_COUNT-1 and c > 0 and c < COLUMN_COUNT-2:
        if board[r-2][c+2] != -turn and board[r-1][c+1] != -turn and board[r+1][c-1] != -turn:
            p_value += (board[r-2][c+2] == turn) + (board[r-1][c+1] == turn) + (board[r][c] == turn) + (board[r+1][c-1] == turn)
    # (r-1,c+1) (r,c) (r+1,c-1) (r+2,c-2) 
    if r > 0 and r < ROW_COUNT-2 and c > 1 and c < COLUMN_COUNT-1:
        if board[r-1][c+1] != -turn and board[r+1][c-1] != -turn and board[r+2][c-2] != -turn:
            p_value += (board[r-1][c+1] == turn) + (board[r][c] == turn) + (board[r+1][c-1] == turn) + (board[r+2][c-2] == turn)
    # (r,c) (r+1,c-1) (r+2,c-2) (r+3,c-3) 
    if r < ROW_COUNT-3 and c > 2:
        if board[r+1][c-1] != -turn and board[r+2][c-2] != -turn and board[r+3][c-3] != -turn:
            p_value += (board[r][c] == turn) + (board[r+1][c-1] == turn) + (board[r+2][c-2] == turn) + (board[r+3][c-3] == turn)
    # (r-3,c) (r-2,c) (r-1,c) (r,c)
    if r > 2:
        if board[r-3][c] != -turn and board[r-2][c] != -turn and board[r-1][c] != -turn:
            p_value += (board[r-3][c] == turn) + (board[r-2][c] == turn) + (board[r-1][c] == turn) + (board[r][c] == turn)
    # (r-2,c) (r-1,c) (r,c) (r+1,c) 
    if r > 1 and r < ROW_COUNT-1:
        if board[r-2][c] != -turn and board[r-1][c] != -turn and board[r+1][c] != -turn:
            p_value += (board[r-2][c] == turn) + (board[r-1][c] == turn) + (board[r][c] == turn) + (board[r+1][c] == turn)
    # (r-1,c) (r,c) (r+1,c) (r+2,c) 
    if r > 0 and r < ROW_COUNT-2:
        if board[r-1][c] != -turn and board[r+1][c] != -turn and board[r+2][c] != -turn:
            p_value += (board[r-1][c] == turn) + (board[r][c] == turn) + (board[r+1][c] == turn) + (board[r+2][c] == turn)
    # (r,c) (r+1,c) (r+2,c) (r+3,c) 
    if r < ROW_COUNT-3:
        if board[r+1][c] != -turn and board[r+2][c] != -turn and board[r+3][c] != -turn:
            p_value += (board[r][c] == turn) + (board[r+1][c] == turn) + (board[r+2][c] == turn) + (board[r+3][c] == turn)
    # (r,c-3) (r,c-2) (r,c-1) (r,c)
    if c > 2:
        if board[r][c-3] != -turn and board[r][c-2] != -turn and board[r][c-1] != -turn:
            p_value += (board[r][c-3] == turn) + (board[r][c-2] == turn) + (board[r][c-1] == turn) + (board[r][c] == turn)
    # (r,c-2) (r,c-1) (r,c) (r,c+1) 
    if c > 1 and c < COLUMN_COUNT-1:
        if board[r][c-2] != -turn and board[r][c-1] != -turn and board[r][c+1] != -turn:
            p_value += (board[r][c-2] == turn) + (board[r][c-1] == turn) + (board[r][c] == turn) + (board[r][c+1] == turn)
    # (r,c-1) (r,c) (r,c+1) (r,c+2) 
    if c > 0 and c < COLUMN_COUNT-2:
        if board[r][c-1] != -turn and board[r][c+1] != -turn and board[r][c+2] != -turn:
            p_value += (board[r][c-1] == turn) + (board[r][c] == turn) + (board[r][c+1] == turn) + (board[r][c+2] == turn)
    # (r,c) (r,c+1) (r,c+2) (r,c+3) 
    if c < COLUMN_COUNT-3:
        if board[r][c+1] != -turn and board[r][c+2] != -turn and board[r][c+3] != -turn:
            p_value += (board[r][c] == turn) + (board[r][c+1] == turn) + (board[r][c+2] == turn) + (board[r][c+3] == turn)
    return p_value

# check if current player wins
def have_winner(board, turn):
    #check columns
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == turn and board[r][c+1] == turn and board[r][c+2] == turn and board[r][c+3] == turn:
                return True
    #check rows
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == turn and board[r+1][c] == turn and board[r+2][c] == turn and board[r+3][c] == turn:
                return True
    #check diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == turn and board[r+1][c+1] == turn and board[r+2][c+2] == turn and board[r+3][c+3] == turn:
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == turn and board[r-1][c+1] == turn and board[r-2][c+2] == turn and board[r-3][c+3] == turn:
                return True
    return False

#check if there's no available move on the board
def no_available_move(borad):
    for c in range(COLUMN_COUNT):
        if board[0][c] == 0:
            return False
    return True

# update interface
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            if (board[r][c] == 0):
                pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2),int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif (board[r][c] == 1):
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2),int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif (board[r][c] == -1):
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2),int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()


board = np.zeros((ROW_COUNT,COLUMN_COUNT), dtype=int)
game_over = False
winner = False
turn = 1
player = 0
# choose player
while player != '1' and player != '2':
    player = input("Choose Player 1 / Player 2 to start the game (1/2): ")
player = int(player)
cur_player = 1

pygame.init()
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        # get mouse position and keep a piece above the board
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0,width,SQUARESIZE))
            posx = event.pos[0]
            if turn == 1:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
            if turn == -1:
                pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()

        # check AI's turn
        if cur_player == 3 - player:
            board = move_with_minimax(board.copy(), turn)
            if no_available_move(board):
                game_over = True
            if have_winner(board, turn):
                game_over = True
                winner = True
            turn = -turn
            print(board)
            draw_board(board)
            #cur_player = 3 - cur_player

        # check play's turn
        if cur_player == player:
            if event.type == pygame.MOUSEBUTTONDOWN:
                finish_move = False
                # player 1 move
                if turn == 1:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))
                    if (is_valid_location(board, col)):
                        board = move_manual(board.copy(), turn, col)
                        finish_move = True
                # player 2 move
                if turn == -1:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))
                    if (is_valid_location(board, col)):
                        board = move_manual(board, turn, col)
                        finish_move = True
                if no_available_move(board):
                    game_over = True
                if have_winner(board, turn):
                    game_over = True
                    winner = True
                if finish_move:
                    turn = -turn
                    print(board)
                    draw_board(board)
                    cur_player = 3 - cur_player
if winner:
    if turn == 1: 
        input ('winner is player 2')
    else:
        input ('winner is player 1')
else:
    input ('game ends in a draw')