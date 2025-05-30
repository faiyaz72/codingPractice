from typing import List
from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root: Node) -> List[List[int]]:
    res = []
    queue = deque([root])
    while len(queue) > 0:
        n = len(queue)
        new_level = []
        for _ in range(n):
            node = queue.popleft()
            new_level.append(node.val)
            for child in [node.left, node.right]:  # enqueue non-null children
                if child is not None:
                    queue.append(child)
        res.append(new_level)
    return res


def binary_tree_right_side_view(root: Node) -> List[int]:
    queue = deque([root])
    res = []
    while len(queue) > 0:
        num_node = len(queue)
        for i in range(num_node):
            node = queue.popleft()
            if i == num_node - 1:
                res.append(node.val)
            for child in [node.left, node.right]:
                if child is not None:
                    queue.append(child)
    return res 

def binary_tree_min_depth(root: Node) -> int:
    queue = deque([root])
    depth = 0
    proceedSearch = True
    while len(queue) > 0 and proceedSearch:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            if node.left is None and node.right is None:
                proceedSearch = False
                break
            for child in [node.left, node.right]:
                if child is not None:
                    queue.append(child)
        depth += 1
    return depth


def shortestPathBinaryMatrix(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """

    def isNeighbour(x, y):
        return grid[x][y] != 1

    def getNeighbour(node: tuple[int, int]):
        neighbourList = []

        # up check
        if (node[0]-1 >= 0 and isNeighbour(node[0]-1, node[1])):
            neighbourList.append((node[0]-1, node[1]))
    
        if (node[0]+1 < len(grid[0]) and isNeighbour(node[0]+1, node[1])):
            neighbourList.append((node[0]+1, node[1]))

        if (node[1]-1 >= 0 and isNeighbour(node[0], node[1]-1)):
            neighbourList.append((node[0], node[1]-1))
        
        if (node[1]+1 < len(grid[0]) and isNeighbour(node[0], node[1]+1)):
            neighbourList.append((node[0], node[1]+1))

        if (node[0]-1 >= 0 and node[1]-1 >= 0 and isNeighbour(node[0]-1, node[1]-1)):
            neighbourList.append((node[0]-1, node[1]-1))
        

        if (node[0]-1 >= 0 and node[1]+1 < len(grid[0]) and isNeighbour(node[0]-1, node[1]+1)):
            neighbourList.append((node[0]-1, node[1]+1))  

        
        if (node[0]+1 < len(grid[0]) and node[1]-1 >= 0 and isNeighbour(node[0]+1, node[1]-1)):
            neighbourList.append((node[0]+1, node[1]-1))  

        if (node[0]+1 < len(grid[0]) and node[1]+1 < len(grid[0]) and isNeighbour(node[0]+1, node[1]+1)):
            neighbourList.append((node[0]+1, node[1]+1))  

        return neighbourList

    row = len(grid)
    column = len(grid[0])
    
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    queue = deque([(0, 0)])
    distance = 1
    visited = set((0,0))
    while len(queue) > 0:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            if node == (row - 1, column - 1):
                return distance
            for neighbour in getNeighbour(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        distance += 1
    
    return -1



def num_steps(target_combo: str, trapped_combos: List[str]) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    # Your task is to determine the least number of moves needed to reach a given target combination from the starting point without hitting any deadend. If reaching the target is impossible due to deadends, return -1.
    
    def getNeighbourSide(current_split,i,isAdd):
        add = 1
        if not isAdd:
            add = -1
        new_one = int(current_split[i]) + add
        if new_one > 9:
            new_one = 0
        elif new_one < 0:
            new_one = 9
        answer_one = current_split[:]
        answer_one[i] = str(new_one)
        return "".join(answer_one)
    
    def getNeighbour(node):
        answer = []
        current_split = list(node)
        for i in range(4):
            toAdd1 = getNeighbourSide(current_split, i, True)
            toAdd2 = getNeighbourSide(current_split,i, False)
            if toAdd1 not in trapped_combos:
                answer.append(toAdd1)
            if toAdd2 not in trapped_combos:
                answer.append(toAdd2)
            
        return answer
    
    
    if (target_combo == '0000'):
        return 0
    
    queue = deque(['0000'])
    visited = set(['0000'])
    distance = 0
    while len(queue) > 0:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            if node == target_combo:
                return distance
            for neighbour in getNeighbour(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        distance +=1
    return -1


def word_ladder(begin: str, end: str, word_list: List[str]) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    def isValid(current: str, word: str) -> bool:
        if (len(current) != len(word)):
            return False
        diff = 0
        for i in range(len(current)):
            if current[i] != word[i]:
                diff += 1
            if diff > 1:
                return False
        
        return True

    def getNeighbour(current: str) -> List[str]:
        answer = []
        for word in word_list:
            if isValid(current, word):
                answer.append(word)
        return answer

    queue = deque([begin])
    visited = set([begin])
    distance = 0
    while len(queue) > 0:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            if node == end:
                return distance
            for neighbour in getNeighbour(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        distance += 1
    return -1


def shortestPath(n: int, edges: List[List[int]], source: int, destination: int) -> int:

    def generateMap():
        adj_map = {}
        for i in range(n):
            adj_map[i] = []
        for edge in edges:
            c_source, c_destination = edge[0], edge[1]
            adj_map[c_source].append(c_destination)
            adj_map[c_destination].append(c_source)
        return adj_map
    
    adjMap = generateMap()
    
    queue = deque([source])
    visited = set([source])
    distance = 0
    while len(queue) > 0:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            if node == destination:
                return distance
            for neighbour in adjMap[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        distance += 1
    return -1

def num_connected_components(n: int, edges: List[List[int]]) -> int:
    def generateMap():
        adj_map = {}
        for i in range(n):
            adj_map[i] = []
        for edge in edges:
            c_source, c_destination = edge[0], edge[1]
            adj_map[c_source].append(c_destination)
            adj_map[c_destination].append(c_source)
        return adj_map
    
    adjMap = generateMap()

    def bfs(source):
        queue = deque([source])
        visited.add(source)
        component = [source]
        while len(queue) > 0:
            num_nodes = len(queue)
            for _ in range(num_nodes):
                node = queue.popleft()
                for neighbour in adjMap[node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
                        component.append(neighbour)
        return component
    visited = set()
    component_list = []
    for node in range(n):
        if node not in visited:
            component_list.append(bfs(node))

    return component_list

def minimum_knight_moves(n, start, target):

    def getNeighbour(node):
        answer = []
        directions = [(2, -1),(2, 1),(-2, 1),(-2, -1),(1,2),(1,-2),(-1,2),(-1,-2)]
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if n > nx >= 0 and n > ny >= 0:
                answer.append((nx, ny)) 

        return answer

    queue = deque([start])
    visited = set([start])
    distance = 0
    while len(queue) > 0:
        num_node = len(queue)
        for _ in range(num_node):
            node = queue.popleft()
            if node == target:
                return distance
            for neighbour in getNeighbour(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        distance += 1
    return -1
    





# this function builds a tree from input; you don't have to modify it
# learn more about how trees are encoded in https://algo.monster/problems/serializing_tree
def build_tree(nodes, f):
    val = next(nodes)
    if val == "x":
        return None
    left = build_tree(nodes, f)
    right = build_tree(nodes, f)
    return Node(f(val), left, right)

if __name__ == "__main__":
    root = build_tree(iter(input().split()), int)
    res = level_order_traversal(root)
    for row in res:
        print(" ".join(map(str, row)))