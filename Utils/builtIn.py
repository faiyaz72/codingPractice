from collections import defaultdict
from typing import List
import heapq
from collections import deque

def convert(s: str):
  return tuple(s.split(":"))

def convert(s: List[str]):
  return " -> ".join(s)

def formatted(stops: List[str], airlines: List[str]):
  ans = []
  for i in range(len(airlines)):
    start = stops[i]
    dst = stops[i + 1]
    ans.append(f"{start} -> {dst} via {airlines[i]}")
  return ans


def find_cheapest_flight(data, source, destination):
  formatted = list(map(lambda x: x.split(":"), data.split(",")))
  map = defaultdict(list)
  for segment in formatted:
    map[segment[0]].append((segment[1], int(segment[3])))
  
  queue = []
  heapq.heappush(queue, (0, source))
  visited = set()
  while len(queue) > 0:
    num_nodes = len(queue)
    for _ in range(num_nodes):
      cost, current = heapq.heappop(queue)
      if current == destination:
        return cost
      if current in visited:
        continue
      visited.add(current)
      for child in map[current]:
        new_cost = cost + child[1]
        heapq.heappush(queue, (new_cost, child[0]))

  return -1

def summarize_transactions(logs: str) -> dict:
  formatted = list(map(lambda x: x.split(":"), logs.split(",")))
  net_spend = defaultdict(int)
  for segment in formatted:
    user, operation, amount = segment
    if operation == "refund" or operation == "chargeback":
      amount = amount * -1
    net_spend[user] = net_spend[user] + amount
  
  return net_spend

def most_frequent_airport(inputList: List[str]) -> str:
  counter = defaultdict(int)
  for segment in inputList:
    parse = segment.split(":")
    counter[parse[0]] += 1
    counter[parse[1]] += 1
  
  currentMax = ""
  maxCount = -1
  for key in counter:
    if counter[key] > maxCount:
      currentMax = key
      maxCount = counter[key]
    
  return currentMax

def remove_duplicates(inputList: List[str]) -> List[str]:
  slow = 0
  for fast in range(len(inputList)):
    if inputList[slow] != inputList[fast]:
      slow += 1
      inputList[slow], inputList[fast] = inputList[fast], inputList[slow]
  return inputList[:slow + 1]

def group(data):
  result = defaultdict(list)
  for origin, dst, airline in data:
    result[origin].append((dst, airline))
  return result

def find_all_routes(data, source, destination):
  adjacencyMap = defaultdict(list)
  for src, dst, airline in data:
    adjacencyMap[src].append((dst, airline))
  
  def dfs(current, path, res, visited: set):
    if current == destination:
      res.append("->".join(path[:]))
      return
    for neighbour in adjacencyMap[current]:
      if neighbour[0] not in visited:
        visited.add(neighbour[0])
        path.append(neighbour[0])
        dfs(neighbour[0], path, res, visited)
        visited.remove(neighbour[0])
        path.pop()
  res = []
  dfs(source, [source], res, set([source]))
  return res

def max_stops(data, src, dst, max_stops):
  adjacencyMap = defaultdict(list)
  for origin, destination, airline in data:
    adjacencyMap[origin].append((destination, airline))
  
  def dfs(current_stop, current_place, res, visited):
    if current_place == dst:
      res.append(current_stop)
      return
    if current_stop > max_stops:
      return
    for neighbour in adjacencyMap[current_place]:
      if neighbour[0] not in visited:
        visited.add(neighbour[0])
        dfs(current_stop + 1, neighbour[0], res, visited)
        visited.remove(neighbour[0])
  res = []
  dfs(0, src, res, set([src]))
  return len(res)

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

def aggregate_price(buy_orders: List[tuple]):
  price_map = defaultdict(int)
  for price, quantity in buy_orders:
    price_map[price]+=quantity
  result = []
  for key, value in price_map.items():
    result.append((key, value))
  return sorted(result, key=lambda x: x[0])

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