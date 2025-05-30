from collections import defaultdict, Counter


def generate_prefix_sum_list(array):
  runningSum = 0
  result = []
  for num in array:
    runningSum = runningSum + num
    result.append(runningSum)
  
  return result


def sum_element_between(array, query):
  prefixSum = [0]
  for num in array:
    prefixSum.append(prefixSum[-1] + num)
  
  result = []
  for seg in query:
    right = seg[1] + 1
    left = seg[0]
    result.append(prefixSum[right] - prefixSum[left])
  
  return result

def if_exist_subarray(array, target):
  prefixMap = {0: -1}
  currentSum = 0
  for i in range(len(array)):
    num = array[i]
    currentSum = currentSum + num
    compliment = currentSum - target
    if compliment in prefixMap:
      return [prefixMap[compliment] + 1, i]
    prefixMap[currentSum] = i
  
  return [-1,-1]


def total_subarray_k(array, target):
  prefixMap = defaultdict(int)
  currentSum = 0
  count = 0
  for i in range(len(array)):
    num = array[i]
    currentSum = currentSum + num
    compliment = currentSum - target
    if compliment in prefixMap:
      count = count + prefixMap[compliment]
    prefixMap[currentSum] = prefixMap[currentSum] + 1
  
  return count

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