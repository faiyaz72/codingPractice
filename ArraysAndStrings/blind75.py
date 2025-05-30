from collections import Counter, defaultdict, deque
from typing import List

def levelOrder(root):
  queue = deque([root])
  ans = []
  while queue:
    num_nodes = len(queue)
    level = []
    for _ in range(num_nodes):
      node = queue.popleft()
      level.append(node.val)
      if node.left:
        queue.append(node.left)
      if node.right:
        queue.append(node.right)
    ans.append(level)
  return ans

def isValidBst(root):
  def helper(root, min, max):
    if root is None:
      return True
    if root.val < min or root.val > max:
      return False
    return helper(root.left, min, root.val) and helper(root.right, root.val, max)
  return helper(root, float('-inf'), float('inf'))

def canConstruct(ransomNote: str, magazine: str):
  magazineMap = defaultdict(int)
  for char in magazine:
    magazineMap[char] = magazineMap[char] + 1
  
  for char in ransomNote:
    if char not in magazineMap:
      return False
    elif magazineMap[char] == 0:
      return False
    else:
      magazineMap[char] = magazineMap[char] - 1 

  return True   


def kClosest(points, k):
  distanceList = []
  for point in points:
    distance = ( (point[0] - 0)**2  + (point[1] - 0)**2  ) ** 0.5
    distanceList.append((point,distance))

  distanceList = sorted(distanceList, key=lambda x: x[1])
  answer = []
  for i in range(k):
    answer.append(distanceList[i][0])
  return answer


def wordBreak(s, wordDict):
  memo = {}
  def dfs(startIndex):
    if startIndex == len(s):
      return True
    if startIndex in memo:
      return memo[startIndex]
    answer = False
    for word in wordDict:
      if s[startIndex:].startswith(word):
        if dfs(startIndex + len(word)):
          answer = True
          break
    
    memo[startIndex] = answer
    return answer




def spiralOrder(matrix):
  def getNext(point: tuple[int, int], prevDirection):
    directions = [(0,1), (1,0), (0, -1), (-1,0)]
    nx, ny = point[0] + prevDirection[0], point[1] + prevDirection[1]
    if len(matrix) > nx >= 0 and len(matrix[0]) > ny >= 0 and matrix[nx][ny] != "#":
        return ((nx, ny), prevDirection)
    for x, y in directions:
      nx, ny = point[0] + x, point[1] + y
      if len(matrix) > nx >= 0 and len(matrix[0]) > ny >= 0 and matrix[nx][ny] != "#":
        return ((nx, ny), (x,y))


  ans = []
  def dfs(point: tuple[int, int], prevDirection):
    ans.append(matrix[point[0]][point[1]])
    matrix[point[0]][point[1]] = "#"
    nextPoint = getNext(point, prevDirection)
    if nextPoint:
      dfs(nextPoint[0], nextPoint[1])
  dfs((0,0), (0,1))
  return ans


def letterCombinations(digits):
  if (digits == ""):
    return []
  answers = []
  letterMap = {
    2: ["a", "b", "c"],
    3: ["d", "e", "f"],
    4: ["g", "h", "i"],
    5: ["j", "k", "l"],
    6: ["m", "n", "o"],
    7: ["p", "q", "r", "s"],
    8: ["t", "u", "v"],
    9: ["w", "x", "y", "z"],
  }

  def dfs(index, seg):
    if len(seg) == len(digits):
      answers.append(seg)
      return
    for letter in letterMap[int(digits[index])]:
      dfs(index + 1, seg + letter)
  
  dfs(0,"")
  return answers

def hasCycle(head):
  map = defaultdict(int)
  current = head
  while current:
    if current.val in map:
      return True
    map[current.val] = 1
    current = current.next
  return False

def binarySearch(nums, target):
  left = 0
  right = len(nums) - 1
  while left <= right:
    mid = (left + right) // 2 
    if (nums[mid] == target):
      return mid
    elif nums[mid] < target:
      left = mid + 1
    else:
      right = mid - 1
  return -1


def eraseOverlapIntervals(intervals):
  sortedList = sorted(intervals, key=lambda x: x[1])
  # [[1,2],[1,3],[2,3],[3,4]]
  slow = 0
  fast = 1
  count = 0
  while fast < len(sortedList):
    if sortedList[fast][0] < sortedList[slow][1]:
      count = count + 1
    else:
      slow = fast
    fast = fast + 1
  
  return count

def replace(x):
  if x == 0:
    return -1
  return x

def findMaxLength(nums):
  formatted = list(map(replace, nums))
  maxIndex = 0
  for i in range(len(formatted)):
    runningSum = runningSum + nums[i]
    if runningSum == 0:
      maxIndex = i
  return maxIndex

def subarraySum(nums, k):
  def dfs(start, total, path, res):
    if total == k:
      res.append(path[:])
      return 
    for i in range(start, len(nums)):
      if total + nums[i] > k:
        continue
      path.append(nums[i])
      dfs(start + 1, total + nums[i], path, res)
      path.pop()
  
  res = []
  dfs(0, 0, [], res)
  return res

def productExceptSelf(nums: List[int]) -> List[int]:
  totalProduct = 1;
  zeroCount = 0
  zeroIndex = -1
  for i in range(len(nums)):
    num = nums[i]
    if (num == 0):
      zeroCount+=1
      zeroIndex = i
      continue
    totalProduct = totalProduct * num

  result = []
  if zeroCount > 1:
    return [0] * len(nums)
  
  for i in range(len(nums)):
    num = nums[i]
    if zeroCount > 0 and i != zeroIndex:
      result.append(0)
    elif i == zeroIndex:
      result.append(totalProduct / 1)
    else:
      result.append(totalProduct / num)

  return result

productExceptSelf([1,2,4,6]);

def topKFrequent(nums: List[int], k:int):
  freqMap = Counter(nums)
  bucket = [-1] * len(nums)
  for num, freq in freqMap.items():
    bucket[freq] = num
  result = []
  count = 0;
  for i in range(len(bucket) - 1, 0, -1):
    if bucket[i] == -1:
      continue
    result.append(bucket[i])
    count += 1
    if count == k:
      return result
  return result

def encode(strs: List[str]) -> str:
  key = ''
  for char in strs:
    key = key + str(len(char)) + "#" + char
  return key

def decode(s: str) -> List[str]:
  result = []
  i = 0
  while i < len(s):
    seg = ''
    strLen = ''
    while s[i] != "#":
      strLen = strLen + s[i]
      i = i + 1
      
    lenSeg = int(strLen)
    for j in range(i + 1, i + lenSeg + 1):
      seg = seg + s[j]
    result.append(seg)
    i = i + lenSeg + 1
  
  return result


def longestConsecutive(nums: List[int]) -> int:
  seenMap = {}
  for i in range(len(nums)):
    if nums[i] not in seenMap:
      seenMap[nums[i]] = i
  
  startNums = set()
  for num in nums:
    if num - 1 not in seenMap:
      startNums.add(num)
  
  maxLength = 0
  for num in startNums:
    currentLen = 0
    currentNum = num
    while currentNum in seenMap:
      currentLen += 1
      currentNum += 1

    maxLength = max(maxLength, currentLen)
  
  return maxLength

def range_sum_query(nums: List[int], left: int, right: int):
  prefixSum = [0]
  for num in nums:
    prefixSum.append(prefixSum[-1] + num)
  
  return prefixSum[right + 1] - prefixSum[left]

def product_of_array_except_self(nums: List[int]):
  left = [1] * len(nums)
  for i in range(len(nums) - 1):
    left[i + 1] = left[i] * nums[i]

  right = [1] * len(nums)
  for i in range(len(nums) - 1,0,-1):
    right[i - 1] =  right[i] * nums[i]
  
  result = []
  for j in range(len(nums)):
    result.append(left[j] * right[j])
  return result

def pivotIndex(nums):

      left = [0]
      right = [0] * (len(nums) + 1)

      for i in range(len(nums)):
        left.append(left[-1] + nums[i])

      for j in range(len(nums), 0, -1):
        right[j - 1] = right[j] + nums[j-1]

      left = left[1:]
      right = right[:-1]

      for i in range(len(left)):
        if left[i] == right[i]:
          return i

      return -1

print(pivotIndex([2,1,-1]))