import heapq
from typing import List
from collections import defaultdict, deque


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

def task_scheduling(tasks: list[str], requirements: list[list[str]]) -> list[str]:

    def buildGraph(requirements: list[list[str]], tasks):
        graph = defaultdict(list)
        for segment in requirements:
            graph[segment[0]].append(segment[1]) 
        
        for node in tasks:
            if node not in graph:
                graph[node] = []
        return graph

    def buildIndgree(graph):
        result = {node: 0 for node in graph}
        for node in graph:
            for child in graph[node]:
                result[child] += 1
        return result


    graph = buildGraph(requirements, tasks)
    in_degree = buildIndgree(graph)

    res = []
    queue = deque()
    for node in in_degree:
        if in_degree[node] == 0:
            queue.append(node)
    while queue:
        node = queue.popleft()
        res.append(node)
        for child in graph[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return res

def task_scheduling_2(tasks: list[str], times: list[int], requirements: list[list[str]]) -> int:
    def buildGraph(requirements: list[list[str]], tasks):
        graph = defaultdict(list)
        for segment in requirements:
            graph[segment[0]].append(segment[1]) 
        
        for node in tasks:
            if node not in graph:
                graph[node] = []
        return graph

    def buildIndgree(graph):
        result = {node: 0 for node in graph}
        for node in graph:
            for child in graph[node]:
                result[child] += 1
        return result
    
    def buildTimeMap(tasks, times):
        result = defaultdict(int)
        for i in range(len(tasks)):
            result[tasks[i]] = times[i]
        return result
    
    graph = buildGraph(requirements, tasks)
    in_degree = buildIndgree(graph)
    time_map = buildTimeMap(tasks, times)

    result = 0
    queue = deque()
    start = {node: 0 for node in tasks}
    for node in in_degree:
            if in_degree[node] == 0:
                queue.append(node)
                start[node] = time_map[node]
                result = max(result, time_map[node])
    while queue:
        node = queue.popleft()
        for child in graph[node]:
            in_degree[child] -= 1
            start[child] = max(start[child], start[node] + time_map[child])
            result = max(start[child], result)
            if in_degree[child] == 0:
                queue.append(child)
    
    return result

tasks = ["a", "b", "c", "d", "e", "f"]
times = [5, 5, 3, 2, 2, 1]
requirements = [["a", "d"], ["b", "d"], ["c", "e"], ["d", "f"], ["e", "f"]]


def service_outage(deps, failed):

    def buildMatrix(deps):
        result = {}
        for segment in deps:
            first_split = segment.split("=")
            result[first_split[0]] = []
            if len(first_split) > 1:
                for value in first_split[1].split("|"):
                    if value != '':
                        result[first_split[0]].append(value)
        return result
    
    adjacency_matrix = buildMatrix(deps)

    def bfs(start, adjacency_matrix):
        child_list = []
        queue = deque([start])
        visited = set()
        visited.add(start)

        while len(queue) > 0:
            for _ in range(len(queue)):
                node = queue.popleft()
                for child in adjacency_matrix[node]:
                    if child not in visited:
                        queue.append(child)
                        visited.add(child)
                        child_list.append(child)
        return child_list

    def max_depth_dfs(node, adjacency_matrix, visited):
        max_depth = 0
        for child in adjacency_matrix.get(node, []):
            if child not in visited:
                visited.add(child)
                depth = 1 + max_depth_dfs(child, adjacency_matrix, visited)
                max_depth = max(max_depth, depth)
                visited.remove(child)  # Allow other paths to use this node (since it's a DAG, not a tree)
        return max_depth


    child_process = bfs(failed, adjacency_matrix)
    dept = max_depth_dfs(failed, adjacency_matrix, set([failed]))

    return (child_process, dept)


def num_steps(target_combo: str, trapped_combos: List[str]) -> int:
    queue = deque(["0000"])
    visited = set(["0000"])
    distance = 1

    def add(node, i):
        temp = node    
        result = int(temp[i]) + 1
        if result > 9:
            result = 0
        temp[i] = str(result)
        return "".join(temp)
    
    def substract(node, i):
        temp = node
        result = int(temp[i]) - 1
        if result < 0:
            result = 9
        temp[i] = str(result)
        return "".join(temp)

    def getNeighbour(node):
        result = []
        for i in range(4):
            afterAdd = add(list(node), i)
            afterSubstract = substract(list(node),i)
            result.append(afterAdd)
            result.append(afterSubstract)
        
        return result

    while len(queue) > 0:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            for child in getNeighbour(node):
                if child == target_combo:
                    return distance
                if child in trapped_combos:
                    continue
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        distance += 1

    return -1


def num_of_islands(grid: list[list[int]]) -> int:

    def getNeighbour(i,j):
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        result = []
        for dx, dy in directions:
            nx, ny = i + dx, j + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
                result.append((nx,ny))
        return result

    def bfs(i,j):
        queue = deque([(i,j)])
        while len(queue) > 0:
            node = queue.popleft()
            for neighbour in getNeighbour(node[0],node[1]):
                grid[neighbour[0]][neighbour[1]] = -1
                queue.append((neighbour[0],neighbour[1]))


    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                result += 1
                grid[i][j] = -1
                bfs(i,j)
    return result


def data_propogation(network, start):

    def buildMap(network):
        result = {}
        for segment in network:
            first_split = segment.split("=")
            key = first_split[0]
            result[key] = []
            values = first_split[1]
            if values != '':
                second_split = values.split("|")
                for value in second_split:
                    third_split = value.split(":")
                    result[key].append((int(third_split[1]),third_split[0]))
        return result


    adjacency_map = buildMap(network)
    # A: [(B,5), (C,3)]

    time_map = {}
    heap = [(0, start)]

    while heap:
        time, node = heapq.heappop(heap)
        if node in time_map:
            continue
        time_map[node] = time
        for child in adjacency_map[node]:
            # child is of type (5,B)
            heapq.heappush(heap, (time + child[0], child[1]))

    return time_map


def network_ttl(edges, start, ttl):

    def buildGraph(edges):
        result = defaultdict(list)
        for src, dst in edges:
            result[src].append(dst)
            result[dst].append(src)
        return result


    graph = buildGraph(edges)
    level = 0
    queue = deque([start])
    visited = set([start])

    node_list = []
    while queue and level <= ttl:
        num_nodes = len(queue)
        for _ in range(num_nodes):
            node = queue.popleft()
            node_list.append(node)
            for child in graph[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        level += 1
    return sorted(node_list)

edges = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("C", "E"),
    ("D", "F")
]
start = "A"
ttl = 2
print(network_ttl(edges, start, ttl))


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

def test_data_propogation():
    # Test 1: Example from your code
    network1 = ["A=B:5|C:3", "B=D:2", "C=D:4", "D="]
    start1 = "A"
    print("Test 1:", data_propogation(network1, start1))  # Expected: {'A': 0, 'B': 5, 'C': 3, 'D': 7}

    # Test 2: Disconnected node
    network2 = ["A=B:2", "B=C:2", "C=", "D="]
    start2 = "A"
    print("Test 2:", data_propogation(network2, start2))  # Expected: {'A': 0, 'B': 2, 'C': 4, 'D': inf} (D unreachable)

    # Test 3: Multiple paths, choose shortest
    network3 = ["A=B:1|C:5", "B=C:1", "C="]
    start3 = "A"
    print("Test 3:", data_propogation(network3, start3))  # Expected: {'A': 0, 'B': 1, 'C': 2}

    # Test 4: Single node
    network4 = ["A="]
    start4 = "A"
    print("Test 4:", data_propogation(network4, start4))  # Expected: {'A': 0}

    # Test 5: Cycle in network
    network5 = ["A=B:1", "B=C:1", "C=A:1"]
    start5 = "A"
    print("Test 5:", data_propogation(network5, start5))  # Expected: {'A': 0, 'B': 1, 'C': 2}

if __name__ == "__main__":
    test_data_propogation()