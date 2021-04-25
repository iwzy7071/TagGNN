class Solution
{
    vector<vector<int>> sums;
public:
    int maxSumSubmatrix(vector<vector<int>> &matrix, int k)
    {
        /*~~~~~~~~~~~~~~~~~~~304 二维区域前缀和~~~~~~~~~~~~~~~~~~~~~~~*/
        int m = matrix.size(), n = matrix[0].size();
        sums = vector<vector<int>>(m + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                sums[i][j] = sums[i][j - 1] + sums[i - 1][j] - sums[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        auto sumRange = [&](int row1, int col1, int row2, int col2) {
            return sums[row2][col2] - sums[row2][col1] - sums[row1][col2] + sums[row1][col1];
        };
        int ans = INT_MIN;
        /*~~~~~~~~~~~~~~~~~~~560 和为k的子数组~~~~~~~~~~~~~~~~~~~~~~~*/
        for (int col1 = 0; col1 < n; col1++) {
            // 用列作外层循环的原因是说明中有 2.如果行数远大于列数 这个条件，测试样例中有这样的样例，
            // 采用行方法也是对的但最后一个样例超时，case by case
            for (int col2 = col1+1; col2 <= n; col2++) {
                set<int> st;
                st.insert(0);
                for (int row = 1; row <= m; row++) {
                    int sum = sumRange(0, col1, row, col2);
                    auto it = st.lower_bound(sum-k);
                    if (it != st.end()) ans = max(ans, sum-*it);
                    st.insert(sum);
                }
            }
        }
        return ans;
    }
};