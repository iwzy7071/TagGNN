class Solution:
    def longestValidParentheses(self, s: str) -> int:
        length = len(s)
        dp = [0] * (length + 1)
        for i in range(1, length + 1):
            if s[i - 1] == ')':
                current = i - 1 - dp[i - 1] - 1
                if current >= 0 and s[current] == '(':
                    dp[i] = dp[i - 1] + 2 + dp[current]

        return max(dp)
