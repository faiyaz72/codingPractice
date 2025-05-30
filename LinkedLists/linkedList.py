import unittest


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        if not self.head:
            self.head = Node(data)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = Node(data)

    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.next
        return count

    def get(self, index):
        current = self.head
        count = 0
        while current:
            if count == index:
                return current.data
            count += 1
            current = current.next
        return None
         

def getDecimalValue(head):
        cur = head
        size = -1
        while cur:
            size += 1
            cur = cur.next

        cur = head
        result = 0
        while cur:
            result = result + (cur.data * 2**size)
            size -= 1
            cur = cur.next
        
        return result

def deleteDuplicates(head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head
        while fast:
            if (fast.data != slow.data):
                slow.next = fast
                slow = fast
            fast = fast.next

def splitListToParts(head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: List[ListNode]
        """
        cur = head
        size = 0
        while cur:
            size += 1
            cur = cur.next

        result = []
        cur = head
        while k > 0:
            if size > 0:
                result.append(cur)
                for i in range(size):
                    cur = cur.next
                size = size - 1
            else:
                result.append(None)
            k -= 1
        return result

def middle_of_linkedList(head: Node):
    slow = head
    fast = head
    while fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def addTwoNumbers(l1, l2):
    strL1 = ""
    strL2 = ""
    hl1 = l1
    hl2 = l2
    while hl1:
            strL1 = str(hl1.data) + strL1
            hl1 = hl1.next
    while hl2:
            strL2 = str(hl2.data) + strL2
            hl2 = hl2.next
    return int(strL1) + int(strL2)

def mergeTwoLists(list1: LinkedList, list2: LinkedList) -> LinkedList:
     mergedList = LinkedList()
     fPointer = list1.head
     sPointer = list2.head
     while fPointer or sPointer:
        if sPointer is None:
            mergedList.insert(fPointer.data)
            fPointer = fPointer.next
        elif fPointer is None:
            mergedList.insert(sPointer.data)
            sPointer = sPointer.next
        elif sPointer.data == fPointer.data:
             mergedList.insert(sPointer.data)
             sPointer = sPointer.next
             fPointer = fPointer.next
        elif sPointer.data < fPointer.data:
             mergedList.insert(sPointer.data)
             sPointer = sPointer.next
        else:
             mergedList.insert(fPointer.data)
             fPointer = fPointer.next

     return mergedList

def has_cycle(nodes: Node) -> bool:
    slow = nodes
    fast = nodes
    while fast and fast.next:
        fast = fast.next.next
        if slow == fast:
            return False
        slow = slow.next
    return True


def removeNthFromEnd(head, n):
  dummy = ListNode(0)
  dummy.next = head
  slow = dummy
  fast = dummy
  for _ in range(n):
      fast = fast.next
  while fast.next:
      fast = fast.next
      slow = slow.next
  slow.next = slow.next.next
  return dummy.next

def detectCycle(head):
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


class TestLinkedList(unittest.TestCase):
    def setUp(self):
        self.list1 = LinkedList()
        self.list2 = LinkedList()

    def test_merge_non_empty_lists(self):
        self.list1.insert(1)
        self.list1.insert(2)
        self.list1.insert(3)
        self.list2.insert(4)
        self.list2.insert(5)
        self.list2.insert(6)
        merged_list = mergeTwoLists(self.list1, self.list2)
        self.assertEqual(merged_list.size(), 6)
        self.assertEqual(merged_list.get(0), 1)
        self.assertEqual(merged_list.get(3), 4)

    def test_merge_non_empty_and_empty_list(self):
        self.list1.insert(1)
        self.list1.insert(2)
        self.list1.insert(3)
        merged_list = mergeTwoLists(self.list1, self.list2)
        self.assertEqual(merged_list.size(), 3)
        self.assertEqual(merged_list.get(0), 1)
        self.assertEqual(merged_list.get(2), 3)

if __name__ == '__main__':
    unittest.main()