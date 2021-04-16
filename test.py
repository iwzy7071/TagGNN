class Solution:
    def isValid(self, s: str) -> bool:
        queue = []
        for ch in s:
            if ch == '(' or ch == '{' or ch == '[':
                queue.append(ch)
            else:
                if len(queue) == 0:
                    return False
                if ch == ')' and queue[-1] == '(':
                    queue.pop()
                elif ch == '}' and queue[-1] == '{':
                    queue.pop()
                elif ch == ']' and queue[-1] == '[':
                    queue.pop()
                else:
                    return False
        return len(queue) == 0