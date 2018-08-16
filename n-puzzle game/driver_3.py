
# coding: utf-8

# In[4]:


import queue as Q
import time
import math
import sys
if sys.platform == "win32":
    import psutil
elif sys.platform == "win64":
    import psutil
else:
    import resource


# In[5]:


#### SKELETON CODE ####

## The Class that Represents the Puzzle
class PuzzleState(object):
    """docstring for PuzzleState"""
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n 
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
        return self.children


# In[6]:


# Function that Writes to output.txt
### Students need to change the method to have the corresponding parameters
def writeOutput(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage):
    ### Student Code Goes here
    text_file = open("Output.txt", "w")
    text_file.write("path_to_goal: %s\n" % path_to_goal)
    text_file.write("cost_of_path: %s\n" % cost_of_path)
    text_file.write("nodes_expanded: %s\n" % nodes_expanded)
    text_file.write("search_depth: %s\n" % search_depth)
    text_file.write("max_search_depth: %s\n" % max_search_depth)
    text_file.write("running_time: %s\n" % running_time)
    text_file.write("max_ram_usage: %s\n" % max_ram_usage)
    text_file.close()
      
def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    path = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = 0
    max_ram_usage = 0
    
    start_time = time.time()
    frontier = Q.Queue()
    frontier.put(initial_state)
    explored = set()
    frontierset = set()
    frontierset.add(initial_state.config)
    
    while not frontier.empty():
        state = frontier.get()
        explored.add(state.config)
        
        if test_goal(state):
            current = state
            while current.parent != None:
                path.append(current.action)
                current = current.parent
            path.reverse()
            running_time = time.time() - start_time
            search_depth = state.cost
            cost_of_path = state.cost
            writeOutput(path, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage)
            return True
        
        nodes_expanded += 1
        
        for neighbor in state.expand():
            if neighbor.config not in explored and neighbor.config not in frontierset:
                frontier.put(neighbor)
                frontierset.add(neighbor.config)
                
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost
                    
        ram = psutil.Process().memory_info().rss / 1024.0 / 1024.0
        if ram > max_ram_usage:
            max_ram_usage = ram
                
    return False
                
def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    path = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = 0
    max_ram_usage = 0
    
    start_time = time.time()
    frontier = []
    frontier.append(initial_state)
    explored = set()
    frontierset = set()
    frontierset.add(initial_state.config)
    
    while frontier:
        state = frontier.pop()
        explored.add(state.config)
        
        if test_goal(state):
            current = state
            while current.parent != None:
                path.append(current.action)
                current = current.parent
            path.reverse()
            running_time = time.time() - start_time
            search_depth = state.cost
            cost_of_path = state.cost
            writeOutput(path, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage)
            return True
        
        nodes_expanded += 1
        
        neighbors = state.expand()
        neighbors.reverse()
        for neighbor in neighbors:
            if neighbor.config not in explored and neighbor.config not in frontierset:
                frontier.append(neighbor)
                frontierset.add(neighbor.config)
                
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost
                    
        ram = psutil.Process().memory_info().rss / 1024.0 / 1024.0
        if ram > max_ram_usage:
            max_ram_usage = ram
                
    return False
    
def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    path = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = 0
    max_ram_usage = 0
    index = 0
    
    start_time = time.time()
    frontier = Q.PriorityQueue()
    frontier.put(comparable(initial_state, index))
    explored = set()
    frontierset = set()
    frontierset.add(initial_state.config)
    
    while frontier.not_empty:
        state = frontier.get().state
        explored.add(state.config)
        
        if test_goal(state):
            current = state
            while current.parent != None:
                path.append(current.action)
                current = current.parent
            path.reverse()
            running_time = time.time() - start_time
            search_depth = state.cost
            cost_of_path = state.cost
            writeOutput(path, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage)
            return True
        
        nodes_expanded += 1
        
        for neighbor in state.expand():
            if neighbor.config not in explored and neighbor.config not in frontierset:
                index += 1
                frontier.put(comparable(neighbor, index))
                frontierset.add(neighbor.config)
                
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost
                    
        ram = psutil.Process().memory_info().rss / 1024.0 / 1024.0
        if ram > max_ram_usage:
            max_ram_usage = ram
                
    return False

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    h = 0
    manhattan_dist = 0
    for i in range(len(state.config)):
        value = state.config[i]
        if value == 0:
            continue
        manhattan_dist = calculate_manhattan_dist(i, value, int(math.sqrt(len(state.config))))
        h = h + manhattan_dist
    
    return (h + state.cost)
    
def calculate_manhattan_dist(idx, value, n):
    """calculatet the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    return (abs(idx // n - value // n) + abs(idx % n - value % n))
    
def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    goalTest = list([])
    for i in range (len(puzzle_state.config)):
        goalTest.append(i)
    goalTest = tuple(goalTest)    
    if puzzle_state.config == goalTest:
        return True
        
class comparable(object):
    def __init__(self, state, index):
        self.state = state
        self.index = index
        self.priority = calculate_total_cost(state)
        
    def __lt__(self, other):
        if self.priority < other.priority:
            return True
        if self.priority == other.priority and self.index < other.index:
            return True
        return False
              
# Main Function that reads in Input and Runs corresponding Algorithm
def main():    
    #begin_state = tuple([6,1,8,4,0,2,7,3,5])
    #sm = 'ast'
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")

if __name__ == '__main__':
    main()

