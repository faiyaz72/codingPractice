from collections import Counter, defaultdict, deque
import heapq
from typing import List


def twoSum(nums, target):
  lookup_map = defaultdict(int)
  result = []
  for i in range(len(nums)):
    compliment = target - nums[i]
    if compliment in lookup_map:
      result.append([lookup_map[compliment], i])
    lookup_map[nums[i]] = i
  
  return result

def threeSum(nums):
  def twoSum(nums, target, append):
    lookup_map = defaultdict(int)
    result = []
    for i in range(len(nums)):
      compliment = target - nums[i]
      if compliment in lookup_map:
        result.append([append, nums[lookup_map[compliment]], nums[i]])
      lookup_map[nums[i]] = i
    
    return result


  result_set = set()
  for i in range(len(nums) - 1):
    target = 0 - (nums[i])
    values = twoSum(nums[i + 1:], target, nums[i])
    if (len(values) > 1):
      for val in values:
        result_set.add(tuple(sorted(val)))
  
  return list(result_set)

def maxArea(heights):
  left = 0
  right = len(heights) - 1
  max_area = 0
  while left < right:
    base = right - left
    height = min(heights[left], heights[right])
    max_area = max(max_area, base * height)
    if heights[left] < heights[right]:
      left += 1
    else:
      right -= 1
  return max_area

def lengthOfLongestSubstring(s):
  length = 0
  left = 0
  seen = defaultdict(int)
  for right in range(len(s)):
    seen[s[right]] += 1
    while seen[s[right]] > 1:
      seen[s[left]] -= 1
      left += 1
    length = max(length, right - left + 1)

def longestOnes(nums, k):
  zeros = 0
  left = 0
  length = 0
  for right in range(len(nums)):
    if nums[right] == 0:
      zeros+=1
    while zeros > k:
      if nums[left] == 0:
        zeros -= 1
      left += 1
    length = max(length, right - left + 1)
  return length


def minSubArrayLen(target, nums):
  length = float("inf")
  total = 0
  left = 0
  for right in range(len(nums)):
    total += nums[right]
    while total >= target:
      length = min(length, right - left + 1)
      total -= nums[left]
      left += 1
    
  
  return length


def groupAnagrams(strs):
  def genKey(data):
    seen = [0] * 26
    a = ord('a')
    for char in data:
      seen[ord(char) - a]+=1
    result = ''
    for num in seen:
      result+=str(num)
    return result
  
  ana_map = defaultdict(list)
  for word in strs:
    ana_map[genKey(word)].append(word)
  
  result = []
  for key, value in ana_map.items():
    result.append(value)
  return result


def lengthOfLongestSubstring(s):
  left = 0
  length = 0
  seen = defaultdict(int)

  for right in range(len(s)):
    seen[s[right]] += 1
    while seen[s[right]] > 1:
      seen[s[left]] -= 1
      left += 1
    length = max(length, right - left + 1)
  return length

def longestConsecutive(nums):
  start = []
  seen = Counter(nums)
  for num in nums:
    if num - 1 not in seen:
      start.append(num)
  
  result = 0
  for num in start:
    current = 1
    check = 1
    while num + check in seen:
      current += 1
      check += 1
    result = max(result, current)
  return result

def search(nums, target):
  """
  :type nums: List[int]
  :type target: int
  :rtype: int
  """
  left = 0 
  right = len(nums) - 1
  pivot_index = 0
  while left <= right:
    mid = (left + right) // 2
    if nums[mid] > nums[-1]:
      left = mid + 1
    else:
      right = mid - 1
      pivot_index = mid

  left = 0
  right = len(nums) - 1
  while left <= right:
    mid = (left + right) // 2
    real_mid = (mid + pivot_index) % len(nums)
    if nums[real_mid] == target:
      return real_mid
    elif nums[real_mid] > target:
      right = mid - 1
    else:
      left = mid + 1

  return -1


def letterCombinations(digits):
    """
    :type digits: str
    :rtype: List[str]
    """

    letter_map = {
      "2": ["a","b","c"],
      "3": ["d","e","f"],
      "4": ["g","h","i"],
      "5": ["j","k","l"],
      "6": ["m","n","o"],
      "7": ["p","q","r","s"],
      "8": ["t","u","v"],
      "9": ["w","x","y","z"]
    }
    if len(digits) == 0:
      return []

    def dfs(start_index, path, res):
      if start_index == len(digits):
        result = "".join(path)
        res.append(result)
        return
      for child in letter_map[digits[start_index]]:
        path.append(child)
        dfs(start_index + 1, path, res)
        path.pop()
    
    res = []
    dfs(0, [], res)
    return res

def find_unmatched_trades(house_trades, street_trades):

  def get_key(segment):
    return f'{segment[0]}-{segment[2]}'
  
  def create_segment_from_trade(trade):
    stripped = trade.lstrip()
    return stripped.split(",")

  def create_map(trades):
    trade_map = defaultdict(list)
    for trade in trades:
      segment = create_segment_from_trade(trade)
      key = get_key(segment)
      trade_map[key].append(trade.lstrip())
    return trade_map
  
  def findIndex(trade, trade_list):
    max_score = 0
    result = (0,-1)
    for i in range(len(trade_list)):
      score = 0
      segment_1 = create_segment_from_trade(trade)
      segment_2 = create_segment_from_trade(trade_list[i])

      if segment_1[0] == segment_2[0] and segment_1[2] == segment_2[2]:
        score = 25
      if segment_1[0] == segment_2[0] and segment_1[2] == segment_2[2] and segment_1[1] == segment_2[1]:
        score = 50
      if trade == trade_list[i]:
        return (100,i)
      

      if score > max_score:
        max_score = score
        result = (score, i)
    return result
  
  def populateTbd(data, score, key, street_trade):
    if key in data:
      data[key].append((score, street_trade))
    else:
      data[key] = [(score, street_trade)]

  
  trade_map = create_map(house_trades)
  

  result = []
  tbd = {defaultdict()}
  for trade in street_trades:
    key = get_key(create_segment_from_trade(trade))
    if key not in trade_map:
      result.append(trade.lstrip())
      continue
    if key in trade_map and trade_map[key]:
      index = findIndex(trade.lstrip(), trade_map[key])
      if index[1] == -1:
        result.append(trade.lstrip())
        continue
      elif index[0] == 100:
        trade_map[key].pop(index[1])
      else:
        populateTbd(tbd[key],index[0],str(index[1]), trade.lstrip())
        tbd[key].append((index[0],str(index[1]), trade.lstrip()))


   
      
  for key, value in trade_map.items():
    for val in trade_map[key]:
      result.append(val)
  return result


house_trades = [
    "GOOG,S,0050,CDC333",
    "AAPL,B,0100,ABC123",
    "TSLA,S,0020,TTT222"
]

street_trades = [
    "GOOG,S,0050,CDC373",
    "GOOG,S,0050,CDC692",
    "AAPL,B,0100,ABC123"
]


def maxProfit(prices):
  current_lowest = float("inf")
  maxProfit = 0
  for num in prices:
    current_lowest = min(current_lowest, num)
    maxProfit = max(num - current_lowest, maxProfit)
  
  return maxProfit

def alertNames(keyName, keyTime):
    """
    :type keyName: List[str]
    :type keyTime: List[str]
    :rtype: List[str]
    """

    def convertToMinutes(timeInStr):
      split = timeInStr.split(":")
      hour = int(split[0])
      minutes = int(split[1])
      return (hour * 60) + minutes

    key_map = defaultdict(list)
    for i in range(len(keyName)):
      key_map[keyName[i]].append(convertToMinutes(keyTime[i]))

    result = []
    for key, val in key_map.items():
      entry_list = sorted(val)
      left = 0
      for right in range(len(entry_list)):
        while entry_list[right] - entry_list[left] > 60:
          left += 1

        if right - left + 1 >= 3:
          result.append(key)
          break
    
    return result

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def removeNthFromEnd(head, n):
  """
  :type head: Optional[ListNode]
  :type n: int
  :rtype: Optional[ListNode]
  """
  dummy = ListNode(0)
  dummy.next = head

  slow = dummy
  fast = dummy

  # Step 1: Move fast n steps ahead
  for _ in range(n):
      fast = fast.next

  # Step 2: Move both until fast reaches the end
  while fast.next:
      fast = fast.next
      slow = slow.next

  # Step 3: Remove the nth node from end
  slow.next = slow.next.next

  return dummy.next
  
def list_to_linked(lst):
    dummy = ListNode(0)
    curr = dummy
    for val in lst:
        curr.next = ListNode(val)
        curr = curr.next
    return dummy.next

def linked_to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def detectCycle(head):
  """
  :type head: ListNode
  :rtype: ListNode
  """

  cycle_found = False
  slow = head
  fast = head

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

    if slow == fast:
      cycle_found = True
      break

  if cycle_found == False:
    return -1
  
  current = slow
  visited = set()
  visited.add(current)
  slow = slow.next
  while slow != current:
    visited.add(slow)
    slow = slow.next
  
  result = 0
  pointer = head
  while pointer not in visited:
    pointer = pointer.next
    result += 1
  
  return result


def create_linked_list_with_cycle(values, pos):
    """
    values: list of node values
    pos: position (0-indexed) where the tail connects to. -1 if no cycle.
    Returns the head of the linked list.
    """
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    nodes = [head]
    for val in values[1:]:
        node = ListNode(val)
        curr.next = node
        curr = node
        nodes.append(node)
    if pos != -1:
        curr.next = nodes[pos]
    return head


def reorderList(head):

  slow = head
  fast = head

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

  reverse_head = reverse_list(slow.next)
  slow.next = None

  pointer = head
  while reverse_head:
    temp_next = pointer.next
    temp_next_reverse = reverse_head.next

    pointer.next = reverse_head
    reverse_head.next = temp_next

    pointer = temp_next
    reverse_head = temp_next_reverse
  
  return head
    



def reverse_list(head):
  prev = None
  current = head
  while current:
    temp_next = current.next
    current.next = prev
    prev = current
    current = temp_next
  
  return prev


def oddEvenList(head):
  """
  :type head: Optional[ListNode]
  :rtype: Optional[ListNode]
  """
  if not head or not head.next:
    return head
  
  current = head
  ref = head.next

  while current and current.next:
    temp = current.next
    current.next = current.next.next
    current = temp
  current.next = ref
  
  return head

def to_linked(lst):
        dummy = ListNode(0)
        curr = dummy
        for val in lst:
            curr.next = ListNode(val)
            curr = curr.next
        return dummy.next

def to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

    # Test case 2: [2,1,3,5,6,4,7] -> [2,3,6,7,1,5,4]
    # Test case 1: [1,2,3,4,5] -> [1,3,5,2,4]
head = to_linked([1,2,3,4,5])
new_head = oddEvenList(head)
print(to_list(new_head))  # Expected: [1,3,5,2,4]


def aggregate_price(buy_orders: List[tuple]):
  price_map = defaultdict(int)
  for price, quantity in buy_orders:
    price_map[price]+=quantity
  
  result = []
  for key, value in price_map.items():
    result.append((key, value))
  
  return sorted(result, key=lambda x: x[0])

def valid_parentheses(s):
  open = {'(': ')', '{': '}', '[': ']'}
  stack = []
  for char in s:
    if char in open:
      stack.append(char)
    else:
      if not stack:
        return False
      corresponding = stack.pop()
      if open[corresponding] != char:
        return False
  return len(stack) == 0


def max_profit(prices: List[int]):
  minimum_seen = float("inf")
  profit = 0
  for price in prices:
    if price > minimum_seen:
      profit = max(price - minimum_seen, profit)
    else:
      minimum_seen = min(minimum_seen, price)
  
  return profit

def merge_sorted_trading_intervals(intervals: List[List[int]], new_interval: List[int]):
  if len(intervals) == 0:
    return [new_interval]

  result = intervals
  first_interval = intervals[0]
  if new_interval[0] < first_interval[0] and new_interval[1] < first_interval[1]:
    result.insert(new_interval)
    return result
  last_interval = intervals[-1]
  if new_interval[0] > last_interval[0] and new_interval[1] > last_interval[1]:
    result.append(new_interval)
    return result
  
  start_index = 0
  end_index = 0
  for i in range(len(intervals)):
    interval = intervals[i]
    if interval[0] <= new_interval[0] <= interval[1]:
      start_index = i
    if interval[0] <= new_interval[1] <= interval[1]:
      end_index = i
  
  # if start and end is same simply update that interval
  if start_index == end_index:
    interval = result[start_index]
    min_amount = min(interval[0], new_interval[0])
    max_amount = max(interval[1], new_interval[1])
    result[start_index] = [min_amount, max_amount]
    return result
  
  # if different, need to merge 
  merged = []
  merge = []
  for i in range(len(result)):
    if i < start_index:
      merged.append(result[i])
    elif i > end_index:
      merged.append(result[i])
    else:
      while start_index <= i <= end_index:
        min_amount = min(result[i][0], new_interval[0])
        max_amount = max(result[i][1], new_interval[1])
        merge = [min_amount, max_amount]
      


def find_kth_largest(nums: list[int], k: int) -> int:
  heap = []
  for i in range(len(nums)):
    heapq.heappush(heap, (nums[i] * -1, i))
  
  index = 0
  while index < k - 1:
    heapq.heappop(heap)
    index += 1
  result, _ = heapq.heappop(heap)
  return -1 * result


def kth_smallest(matrix: list[list[int]], k: int) -> int:
  heap = []
  row = len(matrix)
  for i in range(row):
    heapq.heappush(heap, (matrix[i][0], 0, matrix[i]))
  
  current = 0
  while heap:
    value, head, current_list = heapq.heappop(heap)
    head+=1
    current += 1
    if current == k:
      return value
    if head < len(current_list):
      heapq.heappush(heap, (current_list[head], head, current_list))
  
  return -1

def first_unique_stock(stocks: list[str]):
  counter = defaultdict(int)
  queue = deque([])

  for symbol in stocks:
    counter[symbol] += 1
    queue.append(symbol)
  
  while queue:
    symbol = queue.popleft()
    if counter[symbol] == 1:
      return symbol
  
  return ""

def twoSum(index, nums, target):
  result = []
  complimenet = defaultdict(int)
  for i in range(index+1,len(nums)):
    num = nums[i]
    difference = target - num
    if difference in complimenet:
      compliment_index = complimenet[difference]
      result.append(sorted([nums[index], num, nums[compliment_index]]))
    complimenet[num] = i
  return result

def threeSum(nums):
  result = set()
  for i in range(len(nums)):
    target = 0 - nums[i]
    two_sum_result = twoSum(i, nums, target)
    if len(two_sum_result) > 0:
      for item in two_sum_result:
        result.add(tuple(item))
  
  return list(result)
  
def merged_sorted_linked_list(linkedLists):
  heap = []
  for node in linkedLists:
    heapq.heappush(heap, (node.value, node.next))
  
  dummy = LinkedListNode(-1)
  current = dummy
  while heap:
    value, node = heapq.heappop(heap)
    current.next = node
    current = current.next

    if node.next:
      next_value = node.next.value
      heapq.heappush(heap, (next_value, node.next))

  
  return dummy.next


def order_match(buy_orders, sell_orders):
  min_heap = []
  max_heap = []
  for price,quantity in sell_orders:
    heapq.heappush(min_heap, (price, quantity))

  for price,quantity in buy_orders:
    heapq.heappush(max_heap, (-1 * price, quantity))

  while max_heap and min_heap:
    current_max_price, max_quantity = heapq.heappop(max_heap)
    current_max_price = current_max_price * -1
    current_min_price, min_quanity = heapq.heappop(min_heap)

    if current_max_price >= current_min_price :
      temp_max, temp_min = max_quantity, min_quanity

      max_quantity = max(temp_max - temp_min, 0)
      min_quanity = max(temp_min - temp_max, 0)

      if max_quantity != 0:
        heapq.heappush(max_heap, (current_max_price * -1, max_quantity))
      if min_quanity != 0:
        heapq.heappush(min_heap, (current_min_price, min_quanity))
    else:
      heapq.heappush(max_heap, (current_max_price * -1, max_quantity))
      heapq.heappush(min_heap, (current_min_price, min_quanity))
      break

  
  buy_result = []
  sell_result = []


  while max_heap:
    current_max_price, max_quantity = heapq.heappop(max_heap)
    buy_result.append((current_max_price * -1, max_quantity))
  
  while min_heap:
    current_min_price, min_quanity = heapq.heappop(min_heap)
    sell_result.append((current_min_price, min_quanity))

  return {
    "buy_orders": buy_result,
    "sell_orders": sell_result
  }


def house_robbers(nums):
  if not nums:
    return 0
  
  dp = [0] * len(nums)
  dp[0] = nums[0]

  if len(nums) > 1:
    dp[1] = max(nums[0], nums[1])

  for i in range(2, len(nums)):
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
  
  return dp[-1]


def best_buy_sell_cooldown(prices):
  if not prices:
    return 0

  hold = [0] * len(prices)
  sold = [0] * len(prices)
  rest = [0] * len(prices)

  hold[0] = -prices[0]

  for i in range(1, len(prices)):
    hold[i] = max(hold[i-1], rest[i-1] - prices[i])
    sold[i] = hold[i-1] + prices[i]
    rest[i] = max(rest[i-1], sold[i-1])
  
  return max(sold[-1], rest[-1])

def jump_game(nums):
  if not nums:
    return True
  dp = [False] * (len(nums))
  dp[0] = True
  for i in range(len(nums)):
    if dp[i] == False:
      continue
    if dp[-1] == True:
      return True
    for j in range(1, nums[i] + 1):
      if i + j < len(nums):
        dp[i + j] = True
  
  return dp[-1]

def maximum_subarray(nums):
  dp = [0] * len(nums)
  dp[0] = nums[0]
  for i in range(1, len(nums)):
    dp[i] = max(dp[i-1] + nums[i], nums[i])
  
  return max(dp)

def decode_ways(s):

  if not s or s == "0":
    return -1

  n = len(s) + 1
  dp = [0] * n
  dp[0] = 1
  dp[1] = 1

  for i in range(2, n):
    if s[i-1] != "0":
      dp[i] += dp[i-1]
    two_letter = s[i-2:i]
    if 10 <= int(two_letter) <= 26:
      dp[i] += dp[i-2]
  
  return dp[-1]

def word_break(s, wordDict):
  dp = [False] * len(s)
  dp[0] = True
  for i in range(len(s) + 1):
    for j in range(i):
      if dp[j] and s[j:i] in wordDict:
        dp[i] = True
        break
  return dp[-1]

def extract_unique_ticker(logs):
  '''
  log = "2022-01-01 BUY AAPL 100;2022-01-02 SELL TSLA 50;2022-01-03 BUY AAPL 20;2022-01-04 BUY GOOG 10"
  '''
  ticker_set = set()
  log_list = logs.split(";")
  for log in log_list:
    segment = log.split(" ")
    ticker_set.add(segment[2])
  
  return sorted(list(ticker_set))

def num_unique_path(m: int, n: int):
  dp = [[1] * n for _ in range(m)]
  for i in range(1, m):
    for j in range(1,n):
      dp[i][j] = dp[i-1][j] + dp[i][j-1]
  return dp[m-1][n-1]

def minimal_path_sum(grid: list[list[int]]):
  rows = len(grid)
  columns = len(grid[0])

  dp = [[0] * columns for _ in range(rows)]
  dp[0][0] = grid[0][0]

  for m in range(1, columns):
    dp[0][m] = dp[0][m-1] + grid[0][m]
  
  for n in range(1, rows):
    dp[n][0] = dp[n-1][0] + grid[n][0]
  
  for m in range(1, rows):
    for n in range(1, columns):
      dp[m][n] = min(dp[m-1][n], dp[m][n-1]) + grid[m][n]

  return dp[rows-1][columns-1]

def minimum_total(triangle: list[list[int]]) -> int:
  rows = len(triangle)
  last_row_columns = len(triangle[rows-1])

  dp = []
  for i in range(rows):
    row = []
    for j in range(len(triangle[i])):
      row.append(0)
    dp.append(row)

  # base case, deal with last row
  for i in range(last_row_columns):
    dp[rows-1][i] = triangle[rows-1][i]

  for i in range(rows - 2, -1, -1):
    for j in range(0,len(triangle[i])):
      dp[i][j] = min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle[i][j]
  
  return dp[0][0]

def longest_increasing_sub(nums):
  if not nums:
    return 0

  dp = [0] * len(nums)
  dp[0] = 1

  for i in range(1, len(nums)):
    if (nums[i] > nums[i-1]):
      dp[i] = dp[i-1] + 1
    else:
      dp[i] = dp[i-1]
  
  return dp[len(nums) - 1]

print(longest_increasing_sub([1, 2, 3, 4, 5]))         # Expected: 5 (strictly increasing)
print(longest_increasing_sub([5, 4, 3, 2, 1]))         # Expected: 1 (strictly decreasing)
print(longest_increasing_sub([7, 7, 7, 7]))            # Expected: 1 (all elements equal)
print(longest_increasing_sub([10, 9, 2, 5, 3, 7, 101, 18]))  # Expected: 4
print(longest_increasing_sub([1]))                     # Expected: 1 (single element)
print(longest_increasing_sub([1, 2]))                  # Expected: 2 (two elements, increasing)
print(longest_increasing_sub([2, 1]))                  # Expected: 1 (two elements, decreasing)
print(longest_increasing_sub([1, 3, 2, 4, 3, 5]))      # Expected: 3
print(longest_increasing_sub([1, 2, 2, 3, 4, 1, 5]))   # Expected: 4
print(longest_increasing_sub([]))                      # Edge case: empty list, will raise IndexError