from recbole.model.general_recommender.dmf import DMF


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        dummy_head = ListNode(0)
        curr = dummy_head
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                curr.next = l1
                curr = l1
                l1 = l1.next
            else:
                curr.next = l2
                curr = l2
                l2 = l2.next

        if l1 is not None:
            curr.next = l1
        if l2 is not None:
            curr.next = l2
        return dummy_head.next
