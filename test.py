from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        dummy_head = ListNode(0)
        current = dummy_head

        while True:
            val2index = {}
            for index, head in enumerate(lists):
                if head is None: continue
                val2index[head.val] = index
            if len(val2index) == 0: break

            min_key = min(val2index.keys())
            current.next = lists[val2index[min_key]]
            current = lists[val2index[min_key]]
            lists[val2index[min_key]] = lists[val2index[min_key]].next
        return dummy_head.next


head1 = ListNode(1)
head1.next = ListNode(4)
head1.next.next = ListNode(5)
head2 = ListNode(1)
head2.next = ListNode(3)
head2.next.next = ListNode(4)
head3 = ListNode(2)
head3.next = ListNode(6)

print(Solution().mergeKLists([head1, head2, head3]))
