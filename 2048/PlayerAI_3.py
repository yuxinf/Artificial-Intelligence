import math
import sys
import time
from itertools import product
from BaseAI_3 import BaseAI
from math import log2

class PlayerAI(BaseAI):  
    def getMove(self, grid):
        global start_time
        start_time = time.clock()
        Depth = 0
        initial = state(grid)
        (judge, child, _) = maximize(initial, -math.inf, math.inf, Depth)
        finalAnswer = child
        move = finalAnswer.move
        while True:
            Depth += 1
            (judge, child, _) = maximize(state(grid=grid, depth=Depth), -math.inf, math.inf, Depth)
            
            if Depth > 7:
                break

            #if judge == False:
                #break
                
            finalAnswer = child
            if finalAnswer != None:
                move = finalAnswer.move

        if finalAnswer == None:
            return move
        else:
            move = finalAnswer.move
            return move

def minimize(state, alpha, beta, maxdepth):
    judge = True
        
    if state.depth <= 0:
        return (judge, None, evaluate(state.grid))
    
    children = state.minchildren()
    if len(children) == 0:
        return (judge, None, evaluate(state.grid))
    
    global start_time
    delta_time = time.clock() - start_time
    if delta_time > 0.2:
        judge = False
        return (judge, None, evaluate(state.grid))
        
    (minChild, minUtility) = (None, math.inf)
    
    for child in children:
        (_, _, utility) = maximize(child, alpha, beta, maxdepth)
        
        if children.index(child)%2 == 0:
            utility = utility * 0.9
        if children.index(child)%2 == 1:
            utility = utility * 0.1
        
        if utility < minUtility:
            (minChild, minUtility) = (child, utility)
            
        if minUtility <= alpha:
            break
            
        if minUtility < beta:
            beta = minUtility
            
    return(judge, minChild, minUtility)
                
def maximize(state, alpha, beta, maxdepth):
    judge = True
    if state.depth <= 0:
        return (judge, None, evaluate(state.grid))
    
    children = state.maxchildren()
    if len(children) == 0:
        return (judge, None, evaluate(state.grid))
    
    global start_time
    delta_time = time.clock() - start_time
    if delta_time > 0.2:
        judge = False
        return (judge, None, evaluate(state.grid))
        
    (maxChild, maxUtility) = (None, -math.inf)
    
    for child in children:
        (_, _, utility) = minimize(child, alpha, beta, maxdepth)
        
        if utility > maxUtility:
            (maxChild,maxUtility) = (child, utility)
            
        if maxUtility >= beta:
            break
            
        if maxUtility > alpha:
            alpha = maxUtility
            
    return(judge, maxChild, maxUtility)

class state:
    def __init__(self, grid, move=None, depth=1):
        self.move = move
        self.grid = grid
        self.depth = depth
        
    def maxchildren(self):
        children = []
        for move in self.grid.getAvailableMoves():
            grid = self.grid.clone()
            grid.move(move)
            children.append(state(grid=grid, move=move, depth=self.depth-1))
        return children
        
    def minchildren(self):
        children = []
        cells = self.grid.getAvailableCells()
        iterator = product(cells, [2, 4])
        for cell, tile_value in iterator:
            grid = self.grid.clone()
            grid.setCellValue(cell, tile_value)
            children.append(state(grid=grid, move=self.move, depth=self.depth-1))
        return children

def evaluate(grid):
    weights = [1.0, 2.8, 0.95, 2.72, 2.95]
    #weights = [1.0, 3.0, 0.95, 1.75, 1.7]

    h0 = log2(grid.getMaxTile())
    h1 = len(grid.getAvailableCells())
    h2, h3 = combine_and_smoothness(grid)
    h4 = monotonicity(grid)

    paramters = [h0, h1, h2, h3, h4]

    heuristic = 0
    for w, h in zip(weights, paramters):
        heuristic += w * h

    return heuristic

def monotonicity(grid):
    rows = [0,0,0,0]
    cols = [0,0,0,0]
    diff = 0
 
    for x in range(grid.size):
        for y in range(grid.size-1):
            curr_value = grid.map[x][y]
            next_value = grid.map[x][y+1]

            if next_value != 0: 
                diff = curr_value / next_value
                if diff != 0:
                    diff = log2(diff) 
                else:
                    diff = diff - log2(next_value)
                rows[x] += diff

    for y in range(grid.size):
        for x in range(grid.size-1):
            curr_value = grid.map[x][y]
            next_value = grid.map[x+1][y]

            if next_value != 0: 
                diff = curr_value / next_value
                if diff != 0:
                    diff = log2(diff) 
                else:
                    diff = diff - log2(next_value)
                cols[y] += diff

    return sum(rows) + sum(cols)    

def combine_and_smoothness(grid):
    smooth = 0
    combine = 0
    for x in range(grid.size):
        for y in range(grid.size):
            if grid.map[x][y] != 0:
                value = log2(grid.map[x][y])
                for vector in [(1,0), (0,1)]:
                    l, m = (x,y)
                    i, j = vector
                    target = 0
                    while l < grid.size and m < grid.size and grid.map[l][m] != 0:
                        target = grid.map[l][m]
                        l+=i
                        m+=j
                    if target != 0:
                        smooth -= abs(value - log2(target))

            Y = y+1
            X = x+1
            if Y < grid.size:
                dL = abs(grid.map[x][y] - grid.map[x][Y])
                if dL == 0: 
                    combine += 1
            if X < grid.size:
                dU = abs(grid.map[x][y] - grid.map[X][y])
                if dU == 0: 
                    combine += 1

    return combine, smooth
      
