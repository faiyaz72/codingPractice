from typing import List

def remove_duplicates(arr: List[int]) -> int:
    slow = 0
    fast = 0
    while fast < len(arr):
        if (arr[slow] == arr[fast]):
            fast = fast + 1
        else:
            slow = slow + 1
            arr[slow] = arr[fast]
            fast = fast + 1
            
    
    return slow

def move_zeros(nums: List[int]) -> None:
    slow = 0 
    fast = 0
    while fast < len(nums):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow = slow + 1
        fast = fast + 1  

def buyChoco(prices: List[int], money: int):
    currentMin = money 
    prices.sort()
    end = len(prices) - 1
    while end > 0:
        add = prices[end] + prices[0]
        if (add <= money and money - add < currentMin):
            currentMin = money - add
        else:
            end = end - 1

def two_sum_sorted(arr: List[int], target: int) -> List[int]:
    start = 0
    end = len(arr) - 1
    while end > start:
        addResult = arr[end] + arr[start]
        if (addResult == target):
            return [start, end]
        elif (addResult < target):
            start = start + 1
        else:
            end = end + 1

def isPalindrome(s: str):
    """
    :type s: str
    :rtype: bool
    """
    l = 0
    r = len(s) - 1
    while l < r:
        if not s[l].isalpha:
            l += 1
        if not s[r].isalpha:
            r -= 1
        if (s[l].isalpha and s[r].isalpha):
            if (s[l].lower != s[r].lower):
                return False
            else:
                l += 1
                r -= 1
    
    return True

def maxArea(height):
    """
        You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

        Find two lines that together with the x-axis form a container, such that the container contains the most water.

        Return the maximum amount of water a container can store.
    """
    l = 0
    r = len(height) - 1
    maxArea = 0
    while l < r:
        xDistance = r - l
        currentArea = xDistance * min(height[l], height[r])
        if (currentArea > maxArea):
            maxArea = currentArea
        if (height[r] < height[l]):
            r -= 1
        elif (height[r] > height[l]):
            l += 1
        else:
            r -= 1
            l += 1


    return maxArea

def twoSum(nums: List[int], target: int): 
    left = 0
    right = len(nums) - 1
    sorted = nums.copy()
    sorted.sort()
    sortIndex = []
    while right != left:
        total = sorted[right] + sorted[left]
        if (total == target):
            sortIndex = [left, right]
            break
        elif total < target:
            left += 1
        else:
            right -= 1
    
    resultIndex = []
    for i, val in enumerate(nums):
        if (val == sorted[sortIndex[0]] or val == sorted[sortIndex[1]]):
            resultIndex.append(i)
    
    return resultIndex;

def validParentheses(s: str) -> bool:
    stack = []
    open = {'(': ')', '{': '}', '[': ']'}
    for char in s:
        if char in open:
            stack.append(char)
        elif stack:
            cur = stack.pop()
            if open[cur] != char:
                return False
        else:
            return False
    return len(stack) == 0

def maxProfit(prices: List[int]) -> int:
    maxProfit = 0
    s = 0
    f = 1
    while f < len(prices):
        if (prices[f] < prices[s]):
            s = f        
        else:
            maxProfit = max(prices[f] - prices[s], maxProfit)
        f += 1

    return maxProfit

def subarray_sum_fixed(nums: List[int], k: int) -> int:
    currentSum = sum(nums[:k])
    currentMax = currentSum
    for i in range(k, len(nums)):
        left = i - k
        currentSum -= nums[left]
        currentSum += nums[i]
        currentMax = max(currentMax, currentSum)
    return currentMax

def sort_string(s):
    return ''.join(sorted(s))

def find_all_anagrams(original: str, check: str) -> List[int]:
    result = []
    word = original[0:len(check)]
    sortedWord = sort_string(word)
    if sortedWord == check:
        result.append(0)
    for right in range(len(check), len(original)):
        left = right - len(check)
        word = original[left + 1: right + 1]
        sortedWord = sort_string(word)
        if sortedWord == check:
            result.append(left + 1)
    return result


def canConstruct(ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """

    if len(magazine) < len(ransomNote):
        return False
    ranChar = [0] * 26
    magChar = [0] * 26
    a = ord('a')
    for char in ransomNote:
        ranChar[ord(char) - a] += 1
    for char in magazine:
        magChar[ord(char) - a] += 1
    
    for i in range(len(ranChar)):
        if (ranChar[i] > 0 and magChar[i] < ranChar[i]):
            return False
    return True


def isAnagram(s, t):
    """
    Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    :type s: str
    :type t: str
    :rtype: bool
    """
    charDict = {}
    for char in s:
        if char in charDict:
            charDict[char] += 1
        else:
            charDict[char] = 1
    
    for char in t:
        if char not in charDict:
            return False
        charDict[char] -= 1
        if (charDict[char] == 0):
            del charDict[char]
    
    return len(charDict) == 0

def remove_duplicates_sorted(arr: List[int]):
    slow = 0
    fast = 0
    while fast < len(arr):
        if (arr[fast] != arr[slow]):
            slow += 1
            arr[slow] = arr[fast]
        fast += 1
    return slow + 1

def move_zero(nums: List[int]):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

def container_with_most_water(arr: List[int]):
    start = 0
    end = len(arr) - 1
    max_area = -1 
    while start < end:
        current_area = (end - start) * min(arr[start], arr[end])
        max_area = max(max_area, current_area)
        if (arr[start] >= arr[end]):
            end -= 1
        else:
            start += 1
    return max_area

    
def longest_substring_without_repeating_characters(s: str) -> int:
    counter = {}
    i, j = 0, 0
    longest = 0

    while i < len(s):
        if s[i] not in counter:
            counter[s[i]] = 1
            longest = max(longest, i - j + 1)
            i += 1
        else:
            del counter[s[j]]
            j += 1

    return longest

def num_pairs_divisible_by_60(times: List[int]) -> int:
    answer = 0
    counter = defaultdict(int)
    for num in times:
        compliment = 60 - (num % 60)
        if compliment in counter:
            answer += counter[compliment]
        counter[num % 60] += 1
    return answer

def merge_list(list1, list2): 
    i = 0
    j = 0
    ans = []
    while i < len(list1) and j < len(list2):
        if (list1[i] < list2[j]):
            ans.append(list1[i])
            i+=1
        else:
            ans.append(list2[j])
            j+=1
    if i != len(list1):
        while i < len(list1):
            ans.append(list1[i])
            i += 1
    else:
        while j < len(list2):
            ans.append(list2[j])
            j += 1
    return ans

def can_shift(A, B):
    new = A + A
    return new.find(B) != -1

def find_bigrams(sentence):
    wordList = sentence.split(" ")
    if len(wordList) < 2:
        return []
    i = 0
    j = 1
    ans = []
    while i < len(wordList) and j < len(wordList):
        if i < j:
            ans.append((wordList[i], wordList[j]))
            i = i + 2
        else:
            ans.append((wordList[j], wordList[i]))
            j = j + 2
    return ans

