import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LogicAgent:
    def __init__(self, world):
        self.world = world
        self.visited = set()
        self.safe = set()
        self.unsafe = set()
        self.stench_positions = set()
        self.breeze_positions = set()
        self.wumpus_positions = set()
        self.pit_positions = set()
        self.possible_wumpus = set()
        self.possible_pits = set()
        self.path = []
        self.action_sequence = []
        self.has_planned_path = False
        self.gold_position = None
        self.exit_planned = False
        
        # Initially, the starting position is safe and visited
        self.visited.add((0, 0))
        self.safe.add((0, 0))
    
    def update_knowledge(self):
        x, y = self.world.agent_pos
        percepts = self.world.percepts
        
        # Current cell is safe (since we're in it)
        self.safe.add((x, y))
        self.visited.add((x, y))
        
        # If glitter is perceived, note gold position
        if percepts["glitter"]:
            self.gold_position = (x, y)
        
        # If scream is heard, Wumpus is dead
        if percepts["scream"]:
            self.wumpus_positions.clear()
            self.possible_wumpus.clear()
        
        # Handle stench (possible Wumpus nearby)
        if percepts["stench"]:
            self.stench_positions.add((x, y))
            # Add adjacent unvisited cells as possible Wumpus locations
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.world.grid_size and 
                    0 <= ny < self.world.grid_size and 
                    (nx, ny) not in self.visited and
                    (nx, ny) not in self.safe):
                    self.possible_wumpus.add((nx, ny))
        else:
            # If no stench, adjacent cells cannot have Wumpus
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.world.grid_size and 0 <= ny < self.world.grid_size):
                    if (nx, ny) in self.possible_wumpus:
                        self.possible_wumpus.remove((nx, ny))
                    self.safe.add((nx, ny))
        
        # Handle breeze (possible pit nearby)
        if percepts["breeze"]:
            self.breeze_positions.add((x, y))
            # Add adjacent unvisited cells as possible pit locations
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.world.grid_size and 
                    0 <= ny < self.world.grid_size and 
                    (nx, ny) not in self.visited and
                    (nx, ny) not in self.safe):
                    self.possible_pits.add((nx, ny))
        else:
            # If no breeze, adjacent cells cannot have pits
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.world.grid_size and 0 <= ny < self.world.grid_size):
                    if (nx, ny) in self.possible_pits:
                        self.possible_pits.remove((nx, ny))
                    self.safe.add((nx, ny))
        
        # Deduce Wumpus position if possible
        if len(self.possible_wumpus) == 1:
            w_pos = self.possible_wumpus.pop()
            self.wumpus_positions.add(w_pos)
            self.unsafe.add(w_pos)
        
        # Mark possible pits as unsafe if they can't be anything else
        for pos in self.possible_pits:
            if pos not in self.possible_wumpus:
                self.unsafe.add(pos)
    
    def plan_path_to_safe(self, target=None):
        start = self.world.agent_pos
        start_dir = self.world.agent_dir
        
        if target is None:
            # Find nearest unvisited safe cell
            unvisited_safe = [pos for pos in self.safe if pos not in self.visited]
            if not unvisited_safe:
                return False
            target = unvisited_safe[0]
        
        # BFS to find shortest path
        queue = deque()
        queue.append((start, [], start_dir))
        visited = set()
        visited.add((start, start_dir))
        
        while queue:
            (x, y), path, current_dir = queue.popleft()
            
            if (x, y) == target:
                self.path = path
                self.has_planned_path = True
                return True
            
            # Try all possible moves from current position
            for action in ["move_forward", "turn_left", "turn_right"]:
                new_x, new_y = x, y
                new_dir = current_dir
                new_path = path.copy()
                
                if action == "move_forward":
                    if new_dir == "up":
                        new_x -= 1
                    elif new_dir == "down":
                        new_x += 1
                    elif new_dir == "left":
                        new_y -= 1
                    elif new_dir == "right":
                        new_y += 1
                    
                    # Check if new position is valid and safe
                    if (0 <= new_x < self.world.grid_size and 
                        0 <= new_y < self.world.grid_size and 
                        (new_x, new_y) in self.safe and
                        (new_x, new_y) not in self.unsafe):
                        new_path.append("move_forward")
                        if ((new_x, new_y), new_dir) not in visited:
                            visited.add(((new_x, new_y), new_dir))
                            queue.append(((new_x, new_y), new_path, new_dir))
                else:
                    # Turning changes direction but not position
                    new_path.append(action)
                    if action == "turn_left":
                        dirs = ["up", "left", "down", "right"]
                        idx = dirs.index(new_dir)
                        new_dir = dirs[(idx + 1) % 4]
                    else:  # turn_right
                        dirs = ["up", "right", "down", "left"]
                        idx = dirs.index(new_dir)
                        new_dir = dirs[(idx + 1) % 4]
                    
                    if ((x, y), new_dir) not in visited:
                        visited.add(((x, y), new_dir))
                        queue.append(((x, y), new_path, new_dir))
        
        return False
    
    def decide_action(self):
        self.update_knowledge()
        
        # If we have gold and are at start, exit to win
        if self.world.has_gold and self.world.agent_pos == (0, 0):
            return "exit"
        
        # If we see glitter, grab the gold
        if self.world.percepts["glitter"] and not self.world.has_gold:
            return "grab_gold"
        
        # If we know Wumpus position and have arrow, consider shooting
        if (self.world.has_arrow and 
            len(self.wumpus_positions) == 1 and 
            self.world.wumpus_alive):
            wumpus_pos = next(iter(self.wumpus_positions))
            x, y = self.world.agent_pos
            wx, wy = wumpus_pos
            
            # Check if Wumpus is in line of sight
            if (x == wx and 
                ((self.world.agent_dir == "left" and y > wy) or 
                 (self.world.agent_dir == "right" and y < wy))):
                return "shoot"
            elif (y == wy and 
                  ((self.world.agent_dir == "up" and x > wx) or 
                   (self.world.agent_dir == "down" and x < wx))):
                return "shoot"
        
        # If we have gold, plan path back to start
        if self.world.has_gold and not self.exit_planned:
            if self.plan_path_to_safe((0, 0)):
                self.exit_planned = True
            else:
                # If no path found, try to find one
                return "wait"
        
        # If we have a planned path, follow it
        if self.has_planned_path and self.path:
            action = self.path.pop(0)
            return action
        
        # Otherwise, find a new safe place to explore
        if self.plan_path_to_safe():
            action = self.path.pop(0)
            return action
        
        # If no safe moves, try to shoot Wumpus if we have a good guess
        if (self.world.has_arrow and 
            self.world.wumpus_alive and 
            len(self.possible_wumpus) > 0):
            # Try to face one of the possible Wumpus positions
            for pos in self.possible_wumpus:
                x, y = self.world.agent_pos
                wx, wy = pos
                if x == wx and y < wy and self.world.agent_dir != "right":
                    return "turn_right"
                elif x == wx and y > wy and self.world.agent_dir != "left":
                    return "turn_left"
                elif y == wy and x < wx and self.world.agent_dir != "down":
                    return "turn_right" if self.world.agent_dir == "left" else "turn_left"
                elif y == wy and x > wx and self.world.agent_dir != "up":
                    return "turn_right" if self.world.agent_dir == "right" else "turn_left"
            
            # If facing a possible Wumpus, shoot
            for pos in self.possible_wumpus:
                x, y = self.world.agent_pos
                wx, wy = pos
                if (x == wx and 
                    ((self.world.agent_dir == "left" and y > wy) or 
                     (self.world.agent_dir == "right" and y < wy))):
                    return "shoot"
                elif (y == wy and 
                      ((self.world.agent_dir == "up" and x > wx) or 
                       (self.world.agent_dir == "down" and x < wx))):
                    return "shoot"
        
        # If all else fails, wait (shouldn't happen in a solvable world)
        return "wait"
    
    def execute_action(self, action):
        if action == "move_forward":
            return self.world.move_forward()
        elif action == "turn_left":
            self.world.turn_left()
            return True
        elif action == "turn_right":
            self.world.turn_right()
            return True
        elif action == "shoot":
            return self.world.shoot_arrow()
        elif action == "grab_gold":
            return self.world.grab_gold()
        elif action == "exit":
            return True
        elif action == "wait":
            return True
        return False

class WumpusWorld:
    def __init__(self):
        self.grid_size = 4
        self.agent_pos = (0, 0)  # Starting at (1,1) in grid notation
        self.agent_dir = "right"  # Initial direction
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.world = self.generate_world()
        self.percepts = self.get_percepts()

    def generate_world(self):
        # Initialize empty grid
        world = [[{"pit": False, "wumpus": False, "gold": False} for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        occupied = set()
        occupied.add((0, 0))  # Start cell must remain empty

        # Place Wumpus
        while True:
            wumpus_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if wumpus_pos not in occupied:
                world[wumpus_pos[0]][wumpus_pos[1]]["wumpus"] = True
                occupied.add(wumpus_pos)
                break

        # Place Gold
        while True:
            gold_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if gold_pos not in occupied:
                world[gold_pos[0]][gold_pos[1]]["gold"] = True
                occupied.add(gold_pos)
                break

        # Place pits (20% chance per cell, avoid start and already occupied)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in occupied and random.random() < 0.2:
                    world[i][j]["pit"] = True
                    occupied.add((i, j))  # prevent placing multiple items in same cell

        return world

    def get_percepts(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        percepts = {
            "stench": False,
            "breeze": False,
            "glitter": False,
            "bump": False,
            "scream": False
        }
        
        # Check adjacent cells for Wumpus (stench) and pits (breeze)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.world[nx][ny]["wumpus"] and self.wumpus_alive:
                    percepts["stench"] = True
                if self.world[nx][ny]["pit"]:
                    percepts["breeze"] = True
        
        # Current cell percepts
        if cell["gold"]:
            percepts["glitter"] = True
        
        return percepts

    def move_forward(self):
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if self.agent_dir == "up":
            new_x -= 1
        elif self.agent_dir == "down":
            new_x += 1
        elif self.agent_dir == "left":
            new_y -= 1
        elif self.agent_dir == "right":
            new_y += 1
        
        # Check if move is valid
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)
            self.percepts = self.get_percepts()
            return True
        else:
            self.percepts["bump"] = True
            return False

    def turn_left(self):
        dirs = ["up", "left", "down", "right"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()

    def turn_right(self):
        dirs = ["up", "right", "down", "left"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()

    def shoot_arrow(self):
        if not self.has_arrow:
            return False
        
        self.has_arrow = False
        x, y = self.agent_pos
        wumpus_killed = False
        
        # Arrow travels in a straight line in the current direction
        if self.agent_dir == "up":
            for i in range(x - 1, -1, -1):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "down":
            for i in range(x + 1, self.grid_size):
                if self.world[i][y]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "left":
            for j in range(y - 1, -1, -1):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    break
        elif self.agent_dir == "right":
            for j in range(y + 1, self.grid_size):
                if self.world[x][j]["wumpus"]:
                    wumpus_killed = True
                    break
        
        if wumpus_killed:
            self.wumpus_alive = False
            self.percepts["scream"] = True
            return True
        return False

    def grab_gold(self):
        x, y = self.agent_pos
        if self.world[x][y]["gold"]:
            self.has_gold = True
            self.world[x][y]["gold"] = False
            self.percepts["glitter"] = False
            return True
        return False

    def is_game_over(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        if (cell["pit"] or (cell["wumpus"] and self.wumpus_alive)):
            return "lose"
        if self.has_gold and self.agent_pos == (0, 0):
            return "win"
        return "continue"

    def run_agent(self, max_steps=1000, visualize=False):
        agent = LogicAgent(self)
        steps = 0
        
        if visualize:
            vis = WumpusVisualizer(self)
            vis.draw_world()
        
        while steps < max_steps:
            steps += 1
            action = agent.decide_action()
            
            if action == "exit":
                print("Agent exited with gold! Victory!")
                if visualize:
                    vis.draw_world()
                    plt.show(block=True)
                return "win"
            
            success = agent.execute_action(action)
            
            if not success:
                print("Action failed:", action)
            
            # Check game status
            status = self.is_game_over()
            if status != "continue":
                if status == "win":
                    print("Agent won!")
                else:
                    print("Agent lost!")
                if visualize:
                    vis.draw_world()
                    plt.show(block=True)
                return status
            
            if visualize:
                vis.draw_world()
            
            # Optional: print state for debugging
            print(f"Step {steps}: Pos={self.agent_pos}, Dir={self.agent_dir}, Action={action}")
            print("Percepts:", self.percepts)
        
        print("Max steps reached")
        if visualize:
            vis.draw_world()
            plt.show(block=True)
        return "continue"

class WumpusVisualizer:
    def __init__(self, world):
        self.world = world
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.grid_size = world.grid_size
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # To match grid coordinates
        
    def draw_world(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)
        
        # Draw cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.world.world[i][j]
                color = 'white'
                if cell["pit"]:
                    color = 'black'
                elif cell["wumpus"] and self.world.wumpus_alive:
                    color = 'red'
                elif cell["gold"]:
                    color = 'gold'
                
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                        linewidth=1, edgecolor='gray', 
                                        facecolor=color, alpha=0.5)
                self.ax.add_patch(rect)
                
                # Add text for important cells
                if cell["pit"]:
                    self.ax.text(j, i, "Pit", ha='center', va='center')
                if cell["wumpus"] and self.world.wumpus_alive:
                    self.ax.text(j, i, "Wumpus", ha='center', va='center')
                if cell["gold"]:
                    self.ax.text(j, i, "Gold", ha='center', va='center')
        
        # Draw agent
        x, y = self.world.agent_pos
        dir_symbol = 'v'
        if self.world.agent_dir == "down":
            dir_symbol = '^'
        elif self.world.agent_dir == "left":
            dir_symbol = '<'
        elif self.world.agent_dir == "right":
            dir_symbol = '>'
        
        self.ax.plot(y, x, 'bo', markersize=15)
        self.ax.text(y, x, dir_symbol, ha='center', va='center', color='white')
        
        # Add title with status
        status = ""
        if self.world.has_gold:
            status += "Has Gold "
        if not self.world.has_arrow:
            status += "No Arrow "
        title = f"Wumpus World - Agent at ({x}, {y}) {status}"
        self.ax.set_title(title)
        
        plt.pause(0.5)
    
    def close(self):
        plt.close()

if __name__ == "__main__":
    env = WumpusWorld()
    print("Agent starts at (0, 0). Percepts:", env.percepts)
    result = env.run_agent(visualize=True)
    print("Game result:", result)