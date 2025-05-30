from collections import deque

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def firstBadVersion(self, n):
    """
    :type n: int
    :rtype: int
    """
    left, right = 0, n
    bad = -1
    while left <= right:
        mid = (left + right) // 2
        if (isBadVersion(mid)):
            bad = mid
            right = mid - 1
        else:
            left = mid + 1
    return bad

def binary_tree_right_side_view(root: Node) -> List[int]:
    res = []
    queue = deque([root])
    while len(queue) > 0:
        n = len(queue)
        for i in range(n):
            node = queue.popleft()
            if i == n - 1:
                res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            
    return res

def binary_tree_min_depth(root: Node) -> int:
    queue = deque([root])
    dept = 0
    while len(queue) > 0:
        n = len(queue)
        for _ in range(n):
            node = queue.popleft()
            if node.left is None and node.right is None:
                return dept
            if (node.left):
                queue.append(node.left)
            if (node.right):
                queue.append(node.right)
        dept += 1
    
    return dept