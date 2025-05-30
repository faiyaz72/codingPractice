from collections import defaultdict
from typing import List
import heapq

order = {"country": "CA","items": [{"product": "mouse", "quantity": 20},{"product": "laptop", "quantity": 5}]}


# cost_matrix = {"US": [{"product": "mouse", "cost": 550},{"product": "laptop", "cost": 1000}],"CA": [{"product": "mouse", "cost": 750},{"product": "laptop", "cost": 1100}]}


# calculate_shipping_cost(order_us, shipping_cost) == 16000 // 20 * 550 + 5 * 1000 = 16000
# calculate_shipping_cost(order_ca, shipping_cost) == 20500 // 20 * 750 + 5 * 1100 = 20500


cost_matrix_2 = {
  "US": [
    {
      "product": "mouse",
      "costs": [
        {
          "minQuantity": 0,
          "maxQuantity": None,
          "cost": 550
        }
      ]
    },
    {
      "product": "laptop",
      "costs": [
        {
          "minQuantity": 0,
          "maxQuantity": 2,
          "cost": 1000
        },
        {
          "minQuantity": 3,
          "maxQuantity": 4,
          "cost": 950
        },
        {
          "minQuantity": 5,
          "maxQuantity": None,
          "cost": 900
        }
      ]
    }
  ],
  "CA": [
    {
      "product": "mouse",
      "costs": [
        {
          "minQuantity": 0,
          "maxQuantity": None,
          "cost": 750
        }
      ]
    },
    {
      "product": "laptop",
      "costs": [
        {
          "minQuantity": 0,
          "maxQuantity": 2,
          "cost": 1100
        },
        {
          "minQuantity": 3,
          "maxQuantity": None,
          "cost": 1000
        }
      ]
    }
  ]
}

def getPrice(quantity, cost_matrix):
  total_cost = 0
  remaining_quantity = quantity
  i = 0
  while remaining_quantity:
    min_bracket = cost_matrix[i]["minQuantity"]
    max_bracket = cost_matrix[i]["maxQuantity"]
    cost_bracket = cost_matrix[i]["cost"]


    if max_bracket is None:
      max_bracket = quantity
    if min_bracket == 0:
      min_bracket = 1
    
    available_bracket = max_bracket - min_bracket + 1
    used = min(remaining_quantity, available_bracket)
    total_cost = total_cost + used * cost_bracket 

    remaining_quantity = remaining_quantity - used
    i += 1
  
  return total_cost

def calculate_shipping_cost(order, cost_matrix) -> int:
  dest = order["country"]
  total_cost = 0  
  if (dest not in cost_matrix):
    return -1
  matrix = cost_matrix[dest]
  cost_map = defaultdict(list)
  for product in matrix:
    cost_map[product["product"]] = product["costs"]
  for item in order["items"]:
    product, quantity = item["product"], item["quantity"]
    if product not in cost_map:
      return -1
    price_per_product = getPrice(quantity, cost_map[product])
    total_cost = total_cost + price_per_product

  return total_cost

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

print(calculate_shipping_cost(order, cost_matrix_2))