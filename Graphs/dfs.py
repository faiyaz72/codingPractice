from cmath import inf
from typing import List


def searchDfs(root, target):
    if root is None or root.value == target:
        return root
    return searchDfs(root.left, target) or searchDfs(root.right, target)

def maxValue(root):
    if root is None:
        return -1
    leftMax = maxValue(root.left)
    rightMax = maxValue(root.right)
    return max(root.value, leftMax, rightMax)

def maxDept(root):
    if root is None:
        return 0
    leftMax = maxDept(root.left)
    rightMax = maxDept(root.right)
    return 1 + max(leftMax, rightMax)

def visibleNodeHelper(root, currentMax, numVisibleNode):
    if root is None:
        return 
    if (root.val > currentMax):
        currentMax = root.val
        numVisibleNode[0]+=1
    visibleNodeHelper(root.left, currentMax, numVisibleNode)
    visibleNodeHelper(root.right, currentMax, numVisibleNode)

def visibleNode(root):
    numVisibleNode = [1]
    visibleNodeHelper(root, root.val, numVisibleNode)
    return numVisibleNode[0]

def isBalanced(root):
    if root is None:
        return 0
    left = isBalanced(root.left)
    right = isBalanced(root.right)
    if left == -1 or right == -1:
        return -1
    if abs(left - right) > 1:
        return -1
    return 1 + max(left, right)


def is_same_tree(tree1, tree2):
    if tree1 is None and tree2 is None:
        return True
    if tree1 is None or tree2 is None:
        return False
    check1 = tree1.val == tree2.val
    check2 = is_same_tree(tree1.left, tree2.left)
    check3 = is_same_tree(tree1.right, tree2.right)
    return check1 and check2 and check3

# def subtree_of_another_tree(root: Node, sub_root: Node) -> bool:
#     if not root:
#         return False
#     return is_same_tree(root, sub_root) or subtree_of_another_tree(root.left, sub_root) or subtree_of_another_tree(root.right, sub_root)

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invertBinaryTree(root):
    if root is None:
        return None
    left = invertBinaryTree(root.left)
    right = invertBinaryTree(root.right)
    return Node(root.val, right, left)

def validBst(root):
    return helperValidBst(root, -float('inf'), float('inf'))

def helperValidBst(node, min, max):
    if (node is None):
        return True
    elif (node.val >= max or node.val <= min):
        return False
    else:
        return helperValidBst(node.left, min, node.val) and helperValidBst(node.right, node.val, max)

def insertBst(root, val):
    if root is None:
        return Node(val, None, None)    
    elif root.val > val:
        root.left = insertBst(root.left, val)
    else:
        root.right = insertBst(root.right, val)
    return root

def visible_tree_node(root: Node) -> int:
    def dfs(root, max_sofar):
        if not root:
            return 0

        total = 0
        if root.val >= max_sofar:
            total += 1

        # max_sofar for child node is the larger of previous max and current node val
        total += dfs(root.left, max(max_sofar, root.val))
        total += dfs(root.right, max(max_sofar, root.val))

        return total

    # start max_sofar with smallest number possible so any value root has is smaller than it
    return dfs(root, -inf)

def visible_tree_two(node: Node):

    def dfs(node, max_sofar):
        
        if node is None:
            return 0
        toAdd = 0
        if node.val >= max_sofar:
            toAdd = 1
        max_sofar = max(node.val, max_sofar)
        left = dfs(node.left, max_sofar)
        right = dfs(node.right, max_sofar)
        return right + left + toAdd
    return dfs(node, -inf)


def letter_combination(n: int) -> List[str]:
    def dfs(start_index, path, res):
        if start_index == n:
            res.append("".join(path))
            return
        for char in "ab":
            path.append(char)
            dfs(start_index + 1, path, res)
            path.pop()
    
    res = []
    dfs(0, [], res)


    letMap = {}
    letMap.get()
    return res

def generate_permutations(nums):
    def isLeaf(state):
        return len(state) == len(nums)  # Full permutation reached

    def getEdges(state):
        return [num for num in nums if num not in state]  # Only unused numbers

    def backtrack(state):
        if isLeaf(state):
            result.append(list(state))  # Store valid permutation
            return
        
        for num in getEdges(state):  # Only extend with unused numbers
            state.append(num)
            backtrack(state)  # Recursive step
            state.pop()  # Backtrack to explore other choices

    result = []
    backtrack([])
    return result

def generate_binary_string(n) -> List[str]:
    def dfs(start_index, path, res):
        if (start_index == n):
            res.append("".join(path))
            return
        for edge in "01":
            path.append(edge)
            dfs(start_index + 1, path, res)
            path.pop()
    
    res = []
    dfs(0, [], res)
    return res

def generate_all_seq(n, k) -> List[str]:
    all = 'abcdefghijklmnopqrstuvwxyz'
    subset = all[:k]
    def dfs(start_index, path, res):
        if (start_index == n):
            res.append("".join(path))
            return
        for letter in subset:
            path.append(letter)
            dfs(start_index + 1, path, res)
            path.pop()
    res = []
    dfs(0, [], res)
    return res

def generate_all_roll(n, k):
    def dfs(start_index, path, res):
        if (start_index == k):
            res.append(list(path))
            return
        for i in range(1, n+1):
            path.append(i)
            dfs(start_index + 1, path, res)
            path.pop();
    res = []
    dfs(0, [], res)
    return res

def partition(s: str) -> List[List[str]]:
    # WRITE YOUR BRILLIANT CODE HERE
    def isPalindrome(s):
        return s[::-1] == s

    def dfs(start, path, res):
        if (start == len(s)):
            res.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            prefix = s[start:end]
            if (isPalindrome(prefix)):
                path.append(prefix)
                dfs(end, path, res)
                path.pop()
    res = []
    dfs(0, [], res)
    return res 
    
def combiSum(nums: List[int], target: int):
    def dfs(total, path, res):
        if total == target:
            res.append(path[:])
            return
        for num in nums:
            if total + num > target:
                continue
            path.append(num)
            dfs(total + num, path, res)
            path.pop()
    res = []
    dfs(0, [], res)
    return res 
    
def wordBreak(s):
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    def dfs(start, path, res):
        if start == len(s):
            res.append(" ".join(path))
            return
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word not in wordDict:
                continue
            path.append(word)
            dfs(end, path, res)
            path.pop()
    res = []
    dfs(0, [], res)
    return res 

def restoreIPAddress(s):
    def valid(s: str):
        if len(s) > 3:
            return False
        if int(s) > 255:
            return False
        if len(s) > 1 and s[0] == '0':
            return False
        return True
    
    def dfs(start, path, res):
        if len(path) == 4 and start == len(s):
            res.append(".".join(path[:]))
            return
        if len(path) == 4:
            return
        for end in range(start + 1, len(s) + 1):
            segment = s[start : end]
            if not valid(segment):
                continue
            path.append(segment)
            dfs(end, path, res)
            path.pop()
    res = []
    dfs(0, [], res)
    return res

def permutation(s: List[str]):
    def dfs(path, ignoreState, res):
        if len(path) == len(s):
            res.append(path[:])
            return
        for char in s:
            if char in ignoreState:
                continue
            ignoreState.append(char)
            path.append(char)
            dfs(path, ignoreState, res)
            path.pop()
            ignoreState.remove(char)
    res = []
    dfs([], [], res)
    return res


def combination_without_repeat(nums: List[int], target: int):
    nums.sort()
    def dfs(total, path, res, seenNum):
        if total == target:
            res.append(path[:])
            return
        for i in range(len(nums)):
            if seenNum.get(nums[i], False):  # Check if the element has been seen
                continue
            seenNum[nums[i]] = True 
            path.append(nums[i])
            dfs(total + nums[i], path, res, seenNum)
            path.pop()
            del seenNum[nums[i]]
    res = []
    dfs(0, [], res, {})
    return res




def decode_ways(digits: str) -> int:
    memo = {}

    def isValid(toCheck: str):
        if len(toCheck) == 0 or len(toCheck) > 2:
            return False
        if toCheck[0] == '0':
            return False
        if int(toCheck) >= 1 and int(toCheck) <= 26:
            return True
        return False

    def dfs(startIndex):
        if (startIndex in memo):
            return memo[startIndex]
        if (startIndex == len(digits)):
            return 1
        ans = 0
        for i in range(startIndex + 1, len(digits) + 1):
            if isValid(digits[startIndex:i]):
                ans = ans + dfs(i)
        memo[startIndex] = ans
        return ans
    
    return dfs(0)
    
def coin_change(coins: List[int], amount: int) -> int:
    memo = {}

    def dfs(amount):
        if amount == 0:
            return 0
        if amount in memo:
            return memo[amount]
        ans = inf
        for change in coins:
            if change > amount:
                continue
            result = dfs(amount - change)
            if result != inf:
                ans = min(ans, result + 1)
        memo[amount] = ans
        return ans
    result = dfs(amount)
    return result if result != float('inf') else -1

def minimum_perfect_square(n):
    memo = {}
    def dfs(start):
        if start == 0:
            return 0
        if start in memo:
            return memo[start]
        ans = inf
        for i in range(1,int(start**0.5) + 1):
            if i ** 2 <= start:
                result = dfs(start - i ** 2)
                ans = min(ans, result + 1)
        memo[start] = ans
        return ans
    return dfs(n)

def min_jumps(jump: List[int]):
    memo = {}
    def dfs(start):
        if start == len(jump) - 1:
            return 0
        if start in memo:
            return memo[start]
        ans = inf
        for i in range(1, jump[start] + 1):
            if start + i < len(jump):
                ans = min(ans, dfs(start + i) + 1)
        memo[start] = ans
        return ans
    result = dfs(0)
    return result if result != float('inf') else -1

def detect_cycle(n: int, edges: List[List[int]]) -> bool:
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

    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbour in adjMap[node]:
            if neighbour not in visited:
                if dfs(neighbour, node):
                    return True
            elif neighbour != parent:
                return True
        return False
            
    for node in range(n):
       if node not in visited:
            if dfs(node, -1):
                return True
    return False


def find_path_between_nodes(n: int, edges: List[List[int]], source: int, destination: int) -> List[str]:
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
    visited = set()
    def dfs(node, path, res):
        if node == destination:
            res.append("->".join(path))
            return
        for neighbour in adjMap[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                path.append(str(neighbour))
                dfs(neighbour, path, res)
                path.pop()
                visited.remove(neighbour)
    
    res = []
    dfs(source, [str(source)], res)
    return res

def permutation_str(s: str) -> List[str]:
    def dfs(start, path, res, visited):
        if start == len(s):
            res.append("".join(path))
            return
        for char in s:
            if char not in visited:
                visited.add(char)
                path.append(char)
                dfs(start + 1, path, res, visited)
                visited.remove(char)
                path.pop()
    res = []
    visited = set()
    dfs(0,[],res,visited)
    return res

def subset_sum(nums: List[int], target: int):
    def dfs(path, res):
        if sum(path) == target:
            res.append(path[:])
            return
        for num in nums:
            if num + sum(path) > target:
                continue
            path.append(num)
            dfs(path, res)
            path.pop()
    res = []
    dfs([], res)
    return res

def word_search(board: List[List[str]], word: str) -> bool:

    def getNeighbor(index, board):
        answer = []
        row = len(board)
        column = len(board[0])

        directions = [(0,1), (0,-1), (1,0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = index[0] + dx, index[1] + dy
            if row > new_x >= 0 and column > new_y >= 0 and board[new_x][new_y] != "#":
                answer.append((new_x, new_y))
        return answer


    def dfs(start, board, index):
        if index == len(word):
            return True

        neighbourList = getNeighbor(start, board)
        for i in range(len(neighbourList)):
            currentNeighbour = neighbourList[i]
            letter = board[currentNeighbour[0]][currentNeighbour[1]]
            if (letter != word[index]):
                continue
            board[currentNeighbour[0]][currentNeighbour[1]] = "#"
            if (dfs(currentNeighbour, board, index + 1)):
                return True
            board[currentNeighbour[0]][currentNeighbour[1]] = letter
        return False
    
    for i in range(len(board)):
        for j in range(len(board[i])):
            letter = board[i][j]
            board[i][j] = "#"
            if (dfs((i,j), board, 1)):
                return True
            board[i][j] = letter
    return False


def word_search(board: List[List[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def getNeighbor(index, board):
        answer = []
        directions = [(0,1), (0,-1), (1,0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = index[0] + dx, index[1] + dy
            if 0 <= new_x < rows and 0 <= new_y < cols and board[new_x][new_y] != "#":
                answer.append((new_x, new_y))
        return answer

    def dfs(start, board, index):
        r, c = start

        # Base case: If we have matched all characters
        if index == len(word):
            return True

        # Mark as visited
        temp = board[r][c]
        board[r][c] = "#"

        # Explore all valid neighbors
        for neighbor in getNeighbor((r, c), board):
            nr, nc = neighbor
            if board[nr][nc] == word[index]:  # Correct letter
                if dfs((nr, nc), board, index + 1):
                    return True

        # Backtrack: Restore the original letter
        board[r][c] = temp
        return False

    # Try starting DFS from every cell that matches the first letter
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0]:  # Start only from valid first letters
                if dfs((i, j), board, 1):
                    return True

    return False

def num_islands(grid):
    def getNeighbour(start):
        ans = []
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        for dx, dy in directions:
            nx, ny = start[0] + dx, start[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != "#" and grid[nx][ny] != 2:
                ans.append((nx, ny))
        return ans
    
    def dfs(start):
        grid[start[0]][start[1]] = "#"
        for neighbour in getNeighbour(start):
            dfs(neighbour)
    
    ans = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                ans += 1
                dfs((i,j))
    
    return ans

def find_all_path_between_nodes(graph, src, dst):
    # visited = set([src])
    def dfs(node, path, res, visited):
        if node == dst:
            res.append(path[:])
            return
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                path.append(neighbour)
                dfs(neighbour, path, res, visited)
                visited.remove(neighbour)
                path.pop()

    res = []
    dfs(src,[src],res, set([src]))
    return res

def sum_to_n(integers, N):
    def dfs(total, path, res):
        if total == N:
            res.append(path[:])
            return
        for num in integers:
            if len(path) > 0 and path[-1] > num:
                continue
            if total + num <= N:
                path.append(num)
                dfs(total + num, path, res)
                path.pop()
    
    res = []
    dfs(0, [], res)
    return res


def hasPathSum(root, targetSum):
    def helper(node, currentSum):
        if not node:
            return False
        if currentSum == targetSum:
            return True
        return helper(node.left, currentSum + node.val) or helper(node.right, currentSum + node.val)\
        

def pathSum(root, targetSum):
    def dfs(node, currentSum, path, res):
        if node is None:
            return
        if node.right is None and node.left is None and currentSum == targetSum:
            res.append(path[:])
            return
        for child in [node.left, node.right]:
            if child:
                total = currentSum + child.val
                path.append(child.val)
                dfs(child, total, path, res)
                path.pop()
    res = []
    dfs(root, root.val, [root.val], res)
    return res

def goodNodes(root):
    def helper(node, max_so_far):
        if node is None:
            return 0
        total = 0
        if node.val >= max_so_far:
            total += 1
        new_max = max(max_so_far, node.val)
        return total + helper(node.right, new_max) + helper(node.left, new_max)
    return helper(root, float("-inf"))


class Node:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

def serialize(root):
    result = []
    def dfs(node):
        if node is None:
            result.append("x")
            return
        result.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(result)

def deserialize(code):
    values = code.split(",")
    index = [0]
    def dfs():
        value = values[index[0]]
        if value == "x":
            index[0] += 1
            return None
        index[0]+=1
        left_subtree = dfs()
        right_subtree = dfs()

        return Node(value, left_subtree, right_subtree)
    return dfs()

def print_tree(root, level=0, label="."):
    if root is None:
        print(" " * (level * 2) + label + ": x")
        return
    print(" " * (level * 2) + label + ": " + str(getattr(root, "val", getattr(root, "value", root))))
    print_tree(getattr(root, "left", None), level + 1, "L")
    print_tree(getattr(root, "right", None), level + 1, "R")

def sumOfLeftLeaves(root):
    """
    :type root: Optional[TreeNode]
    :rtype: int
    """
    sum = [0]
    def dfs(node):
        if node is None:
            return
        left = node.left
        if left and left.left is None and left.right is None:
            sum[0] = sum[0] + left.value
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return sum[0]

def maximumRootToLeaf(root):
    maximum = [0]
    def dfs(node, current_sum):
        if not node:
            return None
        current_sum = current_sum + node.val
        if node.left is None and node.right is None:
            maximum[0] = max(maximum[0], current_sum)
            return
        for child in [node.left, node.right]:
            dfs(child, current_sum)
    dfs(root, 0)
    return maximum[0]

def word_exist(board, word):
    row = len(board)
    column = len(board[0])

    def getNeighbour(i, j, index):
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        result = []
        for dx,dy in directions:
            nx, ny = i + dx, j + dy
            if 0 <= nx < row and 0 <= ny < column and board[nx][ny] != "#":
                result.append((nx, ny))
        return result

    def dfs_search(i, j, index):
        if index == len(word):
            return True
        if board[i][j] != word[index]:
            return False
        if index > len(word):
            return False
        temp = board[i][j]
        board[i][j] = "#"
        for neighbour in getNeighbour(i, j, index):
            # getNeighbour returns list of valid and their cordinate
            nx, ny = neighbour
            if (board[nx][ny] != word[index]):
                continue
            if (dfs_search(nx, ny, index + 1)):
                return True
            
        board[i][j] = temp
        return False


    for i in range(row):
        for j in range(column):
            if (dfs_search(i, j, 0)):
                return True
    return False