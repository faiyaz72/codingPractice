from typing import List


def lengthOfLongestSubstring(s):
    """
    Given a string s, find the length of the longest 
    substring
    without repeating characters.
    :type s: str
    :rtype: int
    """
    longest = 0
    window_char = {}
    left = 0
    for right in range(len(s)):
        char = s[right]
        window_char[char] = window_char.get(char, 0) + 1
        while window_char[char] > 1:
            window_char[s[left]] -= 1
            left += 1
        longest = max(longest, right - left + 1)

    return longest

def lengthOfLongestSubstringTwoDistinct(s: str) -> int:
    """
    Given a string, find the length of the longest substring T that contains at most 2 distinct characters.
    """
    longest = 0
    window_char = {}
    left = 0
    for right in range(len(s)):
        char = s[right]
        window_char[char] = window_char.get(char, 0) + 1
        while len(window_char) > 2:
            window_char[s[left]] -= 1
            if window_char[s[left]] == 0:
                del window_char[s[left]]
            left += 1
        longest = max(longest, right - left + 1)
    
    return longest

def subarray_sum_shortest(nums: List[int], target: int) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    window_sum = 0
    min_sum = len(nums)
    left = 0
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum >= target:
            min_sum = min(min_sum, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return min_sum

def least_consecutive_cards_to_match(cards: List[int]) -> int:
    """
    Given a list of integers cards, your goal is to match a pair of cards, 
    but you can only pick up cards in a consecutive manner. What's the minimum number of cards 
    that you need to pick up to make a pair? If there are no matching pairs, return -1.

    For example, given cards = [3, 4, 2, 3, 4, 7], 
    then picking up [3, 4, 2, 3] makes a pair of 3s and picking up [4, 2, 3, 4] 
    matches two 4s. We need 4 consecutive cards to match a pair of 3s and 4 consecutive cards to match 4s, so you need to pick up at least 4 cards to make a match.
    """

    seen_dict = {}
    min_length = float('inf')
    left = 0
    for right in range(len(cards)):
        if (cards[right] in seen_dict):
            seen_dict[cards[right]] += 1
        else:
            seen_dict[cards[right]] = 1
        while seen_dict[cards[right]] == 2:
            min_length = min(min_length, right - left + 1)
            seen_dict[cards[left]] -= 1
            left += 1
    
    if min_length == float('inf'):
        return -1
    return min_length

def prefix_sum(arr: List[int]):
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    return prefix

def subarray_sum(arr: List[int], target: int) -> List[int]:
    """
    Input: arr: 1 -20 -3 30 5 4 target: 7

    Output: 1 3
    """

    left = 0
    window_sum = 0
    for right in range(len(arr)):
        window_sum += arr[right]
        while window_sum > target:
            window_sum -= arr[left]
            left += 1
        if (window_sum == target):
            return [left, right]
    return []

def subarray_sum_total(arr: List[int], target: int) -> int:

    left = 0
    window_sum = 0
    answer = 0
    for right in range(len(arr)):
        window_sum += arr[right]
        while window_sum >= target:
            if (window_sum == target):
                answer += 1
            window_sum -= arr[left]
            left += 1
    return answer

subarray_sum([1, -20, -3, 30, 5, 4], 7)
