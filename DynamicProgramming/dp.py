def climbStairs(n):
  dp = [1, 1]
  for i in range(2, n+1):
    dp.append(dp[i-1] + dp[i-2])
  
  return dp[n]

def tribonacci(n):
  dp = [0, 1, 1]
  for i in range(3, n+1):
    rel = dp[i-1] + dp[i-2] + dp[i-3]
    dp.append(rel)
  return dp[n]

def minCostClimbingStairs(cost):
  dp = [float('inf')] * (len(cost) + 1)
  dp[0] = 0
  dp[1] = 0
  for i in range(2,len(cost) + 1):
    dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
  
  return dp[len(cost)]

def rob(nums):
  dp = [float('-inf')] * (len(nums) + 1)
  dp[0] = nums[0]
  dp[1] = max(nums[0], nums[1])
  for i in range(2, len(nums)):
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
  
  return dp[-1]

rob([2, 1, 1, 9])

def numDecodings(s):

  if not s or s[0] == '0':
    return 0

  dp = [0] * (len(s) + 1)
  dp[0] = 1
  dp[1] = 1
  n = len(s)
  for i in range(2, n + 1):
    one_digit = int(s[i - 1])
    two_digit = int(s[i - 2:i])

    if one_digit >= 1:
      dp[i] += dp[i-1]
    
    if 26 >= two_digit >= 10:
      dp[i] += dp[i-2]

  return dp[n]

numDecodings("11106")

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