# Coding Practice Repository

Welcome to my Coding Practice repository! This project is a curated collection of coding exercises, algorithm implementations, and data structure explorations. The intent is to document my journey in mastering computer science fundamentals, preparing for technical interviews, and demonstrating my problem-solving and coding skills.

## Repository Structure

- **ArraysAndStrings/**: Solutions to classic problems involving arrays and strings, including binary search, sliding window, prefix sums, and two-pointer techniques.
- **DynamicProgramming/**: Dynamic programming problems and solutions, showcasing approaches to optimization and subproblem reuse.
- **Graphs/**: Implementations of graph traversal algorithms such as BFS and DFS, and other graph-related challenges.
- **LinkedLists/**: Linked list operations and problem solutions, demonstrating pointer manipulation and data structure understanding.
- **Utils/**: Utility scripts and playgrounds for experimenting with Python features and built-in functions.
- **InterviewPrep/**: Focused preparation for technical interviews, including curated problems and solutions.

## Highlights

- **Well-Organized**: Each folder targets a specific topic or data structure, making it easy to navigate and find relevant problems.
- **Pythonic Solutions**: All code is written in Python, emphasizing readability, efficiency, and best practices.
- **Algorithmic Variety**: Covers a wide range of algorithms and problem-solving techniques, from brute force to optimized solutions.
- **Continuous Learning**: This repository is regularly updated as I tackle new problems and refine my skills.

## Sample Code

Here is a sample implementation of binary search from the `ArraysAndStrings/binarySearch.py` file:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## How to Use

- Browse the folders to explore solutions to various coding problems.
- Use the code as reference for your own learning or interview preparation.
- Feel free to fork, clone, or contribute!

## License

This repository is for educational purposes. Feel free to use and share the code with attribution.

---

Happy coding!
