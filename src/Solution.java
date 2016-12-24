import java.util.*;

/**
 * Created by zplchn on 11/24/16.
 */
public class Solution {


    //4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null)
            return 0;
        int m = nums1.length, n = nums2.length, sum = m + n;
        if (sum % 2 == 1)
            return getKth(nums1, 0, m -1, nums2, 0, n - 1, (sum/2) + 1);
        else
            return (getKth(nums1, 0, m -1, nums2, 0, n - 1, sum/2) + getKth(nums1, 0, m -1, nums2, 0, n - 1, (sum/2) + 1)) / 2.0;
    }

    private int getKth(int[] nums1, int as, int ae, int[] nums2, int bs, int be, int k){
        int la = ae - as + 1;
        int lb = be - bs + 1;
        if (la > lb)
            return getKth(nums2, bs, be, nums1, as, ae, k);
        if (la == 0)
            return nums2[bs + k - 1];
        if (k == 1)
            return Math.min(nums1[as], nums2[bs]);
        int halfK = Math.min(k/2, la);
        int offA = as + halfK - 1, offB = bs + k - halfK - 1;
        if (nums1[offA] == nums2[offB])
            return nums1[offA];
        else if (nums1[offA] < nums2[offB])
            return getKth(nums1, offA + 1, ae, nums2, bs, offB, k - halfK);
        else
            return getKth(nums1, as, offA, nums2, offB + 1, be, halfK);  //note how k shrink here

    }

    //8
    public int myAtoi(String str) {
        if (str == null)
            return 0;
        str = str.trim();
        if (str.length() == 0)
            return 0;
        int start = 0;
        boolean neg = false;
        if ("+-".indexOf(str.charAt(0)) >= 0){
            start = 1;
            neg = str.charAt(0) == '+'? false: true;
        }
        long res = 0;
        for (int i = start; i < str.length(); ++i){
            if (!Character.isDigit(str.charAt(i)))
                break;
            res = res * 10 + str.charAt(i) - '0';
            if (!neg && res > Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            if (neg && -res < Integer.MIN_VALUE)
                return Integer.MIN_VALUE;
        }
        return (int)(neg? -res: res);

    }

    //9
    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        int div = 1;
        while (x / div >= 10)
            div *= 10;
        while (x != 0){ //here is all the way to x != 0. not x >= 10. like 10021. when remove 1s on the edges, n = 2. will true. but actually should false. here 2 / 10 = 0 != 2.
            if (x / div != x % 10)
                return false;
            x = x % div / 10;
            div /= 100;
        }
        return true;
    }

    //11
    public int maxArea(int[] height) {
        if (height == null || height.length < 2)
            return 0;
        int l = 0, r = height.length - 1, res = 0;
        while (l < r){
            int min = Math.min(height[l], height[r]);
            res = Math.max(res, min * (r - l));
            if (min == height[l])
                ++l;
            else
                --r;
        }
        return res;
    }



    //16
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3)
            return 0;
        Arrays.sort(nums);
        int res = 0, diff = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 2; ++i){
            int l = i + 1, r = nums.length - 1, sum;
            while (l < r){
                sum = nums[i] + nums[l] + nums[r];
                if (Math.abs(sum - target) < diff){
                    diff = Math.abs(target - sum);
                    res = sum;
                }
                if (sum < target)
                    ++l;
                else if (sum > target)
                    --r;
                else
                    break;
            }
        }
        return res;
    }

    //17
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null ||digits.length() == 0)
            return res;
        lettercombiHelper(digits, 0, new StringBuilder(), res);
        return res;
    }
    private final String[] phone = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    private void lettercombiHelper(String digits, int i, StringBuilder sb, List<String> res){
        if (i == digits.length()){
            res.add(sb.toString());
            return;
        }
        String t = phone[digits.charAt(i) - '0'];
        for (int k = 0; k < t.length(); ++k){
            sb.append(t.charAt(k));
            lettercombiHelper(digits, i + 1, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    //20
    public boolean isValid(String s) {
        if (s == null || s.length() == 0)
            return false;
        Deque<Character> st = new ArrayDeque<>();
        for (int i = 0; i < s.length(); ++i){
            if ("([{".indexOf(s.charAt(i)) >= 0)
                st.push(s.charAt(i));
            else {
                if (st.isEmpty())
                    return false;
                char p = st.pop();
                if ((s.charAt(i) == ')' && p != '(')
                    || (s.charAt(i) == ']' && p != '[')
                    || (s.charAt(i) == '}' && p != '{'))
                    return false;
            }
        }
        return st.isEmpty(); // MUST CHECK UNPAIRED LEFT!!!
    }

    //22
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0)
            return res;
        generateHelper(n, 0, 0, "", res);
        return res;
    }

    private void generateHelper(int n, int l, int r, String pre, List<String> res){
        if (r == n){
            res.add(pre);
            return;
        }
        if (l < n)
            generateHelper(n, l + 1, r, pre + "(", res);
        if (r < l) // R < L !!!!
            generateHelper(n, l, r + 1, pre + ")", res);
    }

    //39
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0)
            return res;
        Arrays.sort(candidates);
        combiHelper(candidates, 0, 0, target, new ArrayList<Integer>(), res);
        return res;
    }

    private void combiHelper(int[] candidates, int i, int sum, int target, List<Integer> combi, List<List<Integer>> res){
        if (sum == target){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = i; k < candidates.length; ++k){
            if (k > i && candidates[k] == candidates[k-1])
                continue;
            if (sum + candidates[k] <= target){
                combi.add(candidates[k]);
                combiHelper(candidates, k, sum + candidates[k], target, combi, res);
                combi.remove(combi.size() - 1);
            }
            else
                break;
        }
    }

    //41
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0)
            return 1;
        for (int i = 0; i < nums.length; ++i){
            if (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]){
                int t = nums[nums[i] - 1] ;
                nums[nums[i] - 1] = nums[i];
                nums[i] = t;
                --i;
            }
        }
        for (int i = 0; i < nums.length; ++i){
            if (nums[i] != i + 1)
                return i + 1;
        }
        return nums.length + 1; //here is +1
    }

    //43
    public String multiply(String num1, String num2) {
        if (num1 == null || num1.length() == 0 || num2 == null || num2.length() == 0 || num1.equals("0") ||num2.equals("0"))
            return "0"; //ALL THE FOLLOWING NOT WORKING FOR A NUM IS A 0!!!
        StringBuilder sb1 = new StringBuilder(num1).reverse();
        StringBuilder sb2 = new StringBuilder(num2).reverse();
        int[] tmp = new int[num1.length() + num2.length()];
        for (int i = 0; i < sb1.length(); ++i){
            for (int j = 0; j < sb2.length(); ++j){
                tmp[i + j] += (sb1.charAt(i) - '0') * (sb2.charAt(j) - '0');
            }
        }
        int carry = 0;
        for (int i = 0; i < tmp.length; ++i){
            tmp[i] += carry;
            carry = tmp[i] / 10;
            tmp[i] %= 10;
        }
        StringBuilder res = new StringBuilder();
        for (int i : tmp)
            res.append(i);
        res = res.reverse();
        if (res.charAt(0) == '0')
            return res.substring(1);
        return res.toString();
    }

    //50
    public double myPow(double x, int n) {
        if (n == 0)
            return 1;
        double half = myPow(x, n/2);
        if (n % 2 == 0)
            return half * half;
        else if (n > 0)
            return half * half * x;
        else
            return half * half / x;
    }

    //51
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        if (n <= 0)
            return res;
        queenHelper(n, new int[n], 0, res);
        return res;
    }

    private void queenHelper(int n, int[] columnForQueen, int i, List<List<String>> res){
        if (i ==n){
            List<String> combi = new ArrayList<>(); //here look at the steps
            for (int x: columnForQueen){
                char[] ca = new char[n];
                Arrays.fill(ca, '.');
                ca[x] = 'Q';
                combi.add(new String(ca));
            }
            res.add(combi);
            return;
        }
        for (int k = 0; k < n; ++k){
            columnForQueen[i] = k;
            if (isValidQueen(columnForQueen, i, k))
                queenHelper(n, columnForQueen, i + 1, res);
        }
    }

    private boolean isValidQueen(int[] columnForQueen, int i, int k){
        for (int x = 0; x < i; ++x){
            if (columnForQueen[x] == k || i - x == Math.abs(k - columnForQueen[x]))
                return false;
        }
        return true;
    }

    //58
    public int lengthOfLastWord(String s) {
        if (s == null)
            return 0;
        s = s.trim(); //for String problem, we really need to think if we need trim() in the first place when it's a len >= 1 empty string
        if (s.length() == 0)
            return 0;
        String[] t = s.split("\\s+");
        return t[t.length -1].length();
    }

    //62
    public int uniquePaths(int m, int n) {
        if (m <= 0 || n <= 0)
            return 0;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for (int i = 0; i < dp.length; ++i){
            for (int j = 0; j < dp[0].length; ++j){
                dp[i][j] += (i > 0? dp[i-1][j]: 0) + (j >0? dp[i][j-1]: 0);
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

    //64
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (i == 0 && j == 0)
                    continue;
                else if (i == 0)
                    grid[i][j] += grid[i][j-1];
                else if (j == 0)
                    grid[i][j] += grid[i-1][j];
                else
                    grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    //69
    public int mySqrt(int x) {
        if (x < 0)
            return -1;
        if (x <= 1)
            return x; //here is return x!!
        int l = 1, r = x, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (m < x / m)
                l = m + 1;
            else if (m > x / m)
                r = m - 1;
            else
                return m;
        }
        return r;
    }

    //74
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int l = 0, r = matrix.length - 1, m; //r ---- -1
        while (l < r){
            m = l + ((r - l) >> 1);
            if (target > matrix[m][matrix[0].length - 1])
                l = m + 1;
            else if (target < matrix[m][matrix[0].length - 1])
                r = m;
            else
                return true;
        }

        int row = l;
        l = 0;
        r = matrix[row].length - 1;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (target > matrix[row][m])
                l = m + 1;
            else if (target < matrix[row][m])
                r = m - 1;
            else
                return true;
        }
        return false;
    }

    //77
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < 1 || k < 1)
            return res;
        dfs(n, k, 1, new ArrayList<Integer>(), res);
        return res;
    }

    private void dfs(int n, int k, int start, List<Integer> combi, List<List<Integer>> res){
        if (k ==0){ //better to use less varible
            res.add(new ArrayList<>(combi));
            return;
        }
        if (n - start + 1< k) //this really stops when the number of items left is short than asked.  Prune!
            return;
        for (int j = start; j <= n; ++j){ //Here, j needs to go all way to n to create a combi
            combi.add(j);
            dfs(n, k - 1, j + 1, combi, res);
            combi.remove(combi.size() - 1);
        }
    }

    //79
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0)
            return word.equals("");
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                if (board[i][j] == word.charAt(0))
                    if (existHelper(board, i, j, word, 0))
                        return true;
            }
        }
        return false;
    }
    private final int[][] eoff = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    private boolean existHelper(char[][] board, int i, int j, String word, int k){//FOR CHAR, use the invalid to tell. Int filter before entering children
        if (k == word.length())
            return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(k) || (board[i][j] & 256) != 0)
            return false;

        board[i][j] ^= 256;
        boolean res = false;
        for (int z = 0; z < eoff.length; ++z){
            int x = i + eoff[z][0], y = j + eoff[z][1];
            if ((res = existHelper(board, x, y, word, k+1)) == true)
                break;
        }
        board[i][j] ^= 256;
        return res;
    }

    //86
    public ListNode partition(ListNode head, int x) {
        if (head == null)
            return head;
        ListNode ds = new ListNode(0), ps = ds;
        ListNode dg = new ListNode(0), pg = dg;
        ListNode cur = head;
        while (cur != null){
            if (cur.val < x){
                ps.next = cur;
                ps = ps.next;
            }
            else {
                pg.next = cur;
                pg = pg.next;
            }
            cur = cur.next;
        }
        pg.next = null; //Here, when merge linkedlist, must check set null to the end!!
        ps.next = dg.next;
        return ds.next;
    }

    //90
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        Arrays.sort(nums);
        res.add(new ArrayList<>());
        int size = 1, start = 0;
        for (int i = 0; i < nums.length; ++i){
            if (i > 0 && nums[i] == nums[i-1])
                start = size;
            else
                start = 0; //Here!!!!!
            size = res.size();
            for (int j = start; j < size; ++j){
                List<Integer> l = new ArrayList<>(res.get(j));
                l.add(nums[i]);
                res.add(l);
            }
        }
        return res;
    }

    //93
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 4)
            return res;
        dfs(s, 0, 0, "", res);
        return res;
    }

    private void dfs(String s, int i, int p, String pre, List<String> res){
        if (p == 3){
            String rest = s.substring(i);
            if (isValidip(rest))
                res.add(pre + rest);
            return;
        }
        for (int k = i + 1; k < s.length() && k <= i + 3; ++k){ //HERE, dont forget to always check < length() before check <= i + 3!!
            String str = s.substring(i, k);
            if (isValidip(str))
                dfs(s, k, p + 1, pre + str + ".", res);
        }

    }

    private boolean isValidip(String s){
        return s.length() > 0 && s.length() <= 3 && (s.length() > 1? s.charAt(0) != '0': true) && Integer.parseInt(s) <= 255;
    }

    //94
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        while (!st.isEmpty() || root != null){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                TreeNode tn = st.pop();
                res.add(tn.val);
                root = tn.right;
            }
        }
        return res;
    }

    //99
    private TreeNode pre;
    public void recoverTree(TreeNode root) {
        if (root == null)
            return;
        TreeNode[] rev = new TreeNode[2];
        dfs(root, rev);
        int t = rev[0].val;
        rev[0].val = rev[1].val;
        rev[1].val = t;
    }

    private void dfs(TreeNode root, TreeNode[] rev){
        if (root == null)
            return;
        dfs(root.left, rev);
        if (pre != null && root.val < pre.val){
            if (rev[0] == null) //!!
                rev[0] = pre;
            rev[1] = root;
        }
        pre = root;
        dfs(root.right, rev);
    }

    //102
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int cur = 1, next = 0;
        List<Integer> combi = new ArrayList<>();
        while (!queue.isEmpty()){
            TreeNode tn = queue.poll();
            combi.add(tn.val);
            if (tn.left != null){
                queue.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                queue.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                res.add(combi);
                combi = new ArrayList<>();
                cur = next;
                next = 0;
            }
        }
        return res;
    }

    //104
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //111
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null)
            return minDepth(root.right) + 1;
        if (root.right == null)
            return minDepth(root.left) + 1;
        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

    //128
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        Set<Integer> hs = new HashSet<>();
        for (int i : nums)
            hs.add(i);
        int res = 1;
        do {
            Iterator<Integer> iter = hs.iterator();
            int x = iter.next();
            hs.remove(x); //DONT FORGET REMVOE ITSELF!
            int l = x - 1, r = x + 1, len = 1;
            while (hs.contains(l)){
                ++len;
                hs.remove(l--);
            }
            while (hs.contains(r)){
                ++len;
                hs.remove(r++);
            }
            res = Math.max(res, len);
        } while (!hs.isEmpty());
        return res;
    }

    //130
    public void solve(char[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0)
            return;
        //start at 4 edges, bfs and change 'O' to '#'. at the end, 'O' to 'X', '#' back to 'O'
        for (int j = 0; j < board[0].length; ++j){
            if (board[0][j] == 'O')
                floodFill(board, 0, j);
            if (board[board.length - 1][j] == 'O')
                floodFill(board, board.length - 1, j);
        }
        for (int i = 0; i < board.length; ++i){
            if (board[i][0] == 'O')
                floodFill(board, i, 0);
            if (board[i][board[0].length - 1] == 'O')
                floodFill(board, i, board[0].length - 1);
        }
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
                if (board[i][j] == '#')
                    board[i][j] = 'O';
            }
        }
    }
    private final int[][] foff = {{-1, 0},{1, 0},{0, -1},{0, 1}};
    private void floodFill(char[][] board, int i, int j){
        board[i][j] = '#';
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{i, j});
        while (!queue.isEmpty()){
            int[] t = queue.poll();
            for (int k = 0; k < foff.length; ++k){
                int x = t[0] + foff[k][0], y = t[1] + foff[k][1];
                if (x >= 0 && x < board.length && y >= 0 && y < board[0].length && board[x][y] == 'O'){
                    board[x][y] = '#';
                    queue.offer(new int[]{x, y});
                }
            }
        }
    }

    //131
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        if (s == null || s.length() == 0)
            return res;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1]))
                    dp[i][j] = true;
            }
        }
        partitionHelper(s, 0, dp, new ArrayList<String>(), res);
        return res;
    }

    private void partitionHelper(String s, int i, boolean[][] dp, List<String> combi, List<List<String>> res){
        if (i == s.length()){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int j = i + 1; j <= s.length(); ++j){
            if (dp[i][j-1]){
                combi.add(s.substring(i, j));
                partitionHelper(s, j, dp, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //134
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null || gas.length == 0 || gas.length != cost.length)
            return 0;
        int local = 0, total = 0, start = 0;
        for (int i = 0; i < gas.length; ++i){
            int add = gas[i] - cost[i];
            local += add; //here must update local first!!!
            total += add;
            if (local< 0){ //not = 0 as long as we can arrive at next station [2] [2] enough to arr next station
                local = 0;
                start = i + 1;
            }

        }
        return total >= 0? start: -1;

    }

    //146
    public class LRUCache {

        class ListNode{
            int val, key;
            ListNode next, pre;
            ListNode(){}
            ListNode(int k, int v){
                key = k;
                val = v;
            }
        }

        private Map<Integer, ListNode> hm; //Note all private
        private ListNode head, tail;
        private int cap, size;

        public LRUCache(int capacity) {
            hm = new HashMap<Integer, ListNode>();
            head = new ListNode();
            tail = new ListNode();
            head.next = tail;
            tail.pre = head;
            cap = capacity;
        }

        public int get(int key) {
            if (!hm.containsKey(key))
                return -1;
            ListNode ln = hm.get(key);
            moveToHead(ln);
            return ln.val;
        }

        public void set(int key, int value) {
            ListNode ln;
            if (hm.containsKey(key)){
                ln = hm.get(key);
                ln.val = value;
                moveToHead(ln);
            }
            else {
                if (size == cap){
                    ln = tail.pre;
                    hm.remove(ln.key);
                    removeNode(ln);
                    --size; //Dont forget adjust size
                }
                ln = new ListNode(key, value);
                hm.put(key, ln);
                insertToHead(ln);
                ++size;//Dont forget adjust size
            }
        }

        private void moveToHead(ListNode ln){
            if (head.next == ln)
                return;
            removeNode(ln);
            insertToHead(ln);
        }

        private void insertToHead(ListNode ln){
            ln.next = head.next;
            head.next.pre = ln;
            head.next = ln;
            ln.pre = head;
        }

        private void removeNode(ListNode ln){
            ln.pre.next = ln.next;
            ln.next.pre = ln.pre;
        }


    }

    //155
    public class MinStack {
        Deque<Integer> st;
        Deque<Integer> mt;
        /** initialize your data structure here. */
        public MinStack() {
            st = new ArrayDeque<>();
            mt = new ArrayDeque<>();
        }

        public void push(int x) {
            st.push(x);
            if (mt.isEmpty() || x <= mt.peek())
                mt.push(x);
        }

        public void pop() {
            if (st.pop().equals(mt.peek())) //Note here must use equals(). Reason is it's an Integer and two Integer object must use equals to compare. if use ==, will not true !!!!!
                mt.pop();
        }

        public int top() {
            return st.peek();
        }

        public int getMin() {
            return mt.peek();
        }
    }

    //157
    int read4(char[] buf){return 0;}

    public int read(char[] buf, int n) {
        //the problem is , call read4 and output to the buf char[]
        if (buf == null || buf.length == 0 || n <= 0)
            return 0;
        char[] buf4 = new char[4];
        boolean eof = false;
        int idx = 0;
        while (!eof && n > 0){
            int x = Math.min(n, read4(buf4));
            if (x < 4)
                eof = true;
            n -= x;
            System.arraycopy(buf4, 0, buf, idx, x); //System.arraycopy(int[] src, int srcidx, int[] target, int taridx, int len)
            idx += x;
        }
        return idx;
    }

    //158
    private char[] buffer = new char[4];
    private int bufCnt, bufIdx;

    public int read2(char[] buf, int n) {
        if (buf == null || buf.length < n || n <= 0)
            return 0;
        boolean eof = false;
        int i = 0;
        while (!eof && n > 0){
            if (bufCnt == 0){
                bufCnt = read4(buffer);
                bufIdx = 0;
                if (bufCnt < 4)
                    eof = true;
            }
            int x = Math.min(bufCnt, n);
            System.arraycopy(buffer, bufIdx, buf, i, x);
            i += x;
            n -= x;
            bufIdx = (bufIdx + x) % 4; //4 conditions to do, src, target, buf*2
            bufCnt -=x;
        }
        return i;

    }

    //159
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        int l = 0, r = 0, res = 0;
        while (r < s.length()){
            //1.read in (move right)
            hm.put(s.charAt(r), hm.getOrDefault(s.charAt(r), 0) + 1);
            //2. adjust (shrink left)
            while (hm.size() > 2){
                if (hm.get(s.charAt(l)) == 1)
                    hm.remove(s.charAt(l));
                else
                    hm.put(s.charAt(l), hm.get(s.charAt(l)) - 1);
                ++l;
            }
            //3. get max
            res = Math.max(res, r - l + 1);
            ++r;
        }
        return res;
    }

    //163
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0){
            //res.add(lower + "->" + upper);
            res.add(missingHelper(lower, upper)); //when nums is empty, lower can == upper
            return res;
        }
        if (nums[0] > lower)
            res.add(missingHelper(lower, nums[0] - 1));
        for (int i = 1; i < nums.length; ++i){
            //if (nums[i]  - nums[i-1] > 1)
            if (nums[i] > nums[i-1] + 1) //if nums[i-1] is -INF, and nums[i] is INF. can only use +. not - overflow!!
                res.add(missingHelper(nums[i-1] + 1, nums[i] - 1));
        }
        if (upper > nums[nums.length - 1])
            res.add(missingHelper(nums[nums.length - 1] + 1, upper));
        return res;
    }

    private String missingHelper(int s, int e){
        if (s == e)
            return Integer.toString(s);
        else
            return s + "->" + e;
    }

    //166
    public String fractionToDecimal(int numerator, int denominator) {
        /*our thought is first get the integral part. and r = n / d is the remainder which is always less than d
        at each time, we have r. and we set nr = r * 10, val = nr / d, new r = nr % d;
        the key is r can repeat appear, so the val will be repeated as well. so we need to set a map regarding r - offset
        we reoccrence of r happens, we seperate the non-reoccur part and add () around the reoccur part

        test cases: +/-. use long as we r * 10 when INF - 1 % INF --> INF - 1 --> *10 will overflow; when eitehr is 0 return 0

         */
        if (numerator == 0 || denominator == 0)
            return "0"; //must use >>>
        boolean isNeg = ((numerator ^ denominator) >>> 31) == 1; // this must be done before convert to long otehrwise the sign bit will go to 63th
        long n = numerator, d = denominator;
        n = Math.abs(n);
        d = Math.abs(d);
        long integ = n / d, r = n % d;
        if (r == 0)
            return (isNeg? "-": "") + integ; //must use () here !!
        //now we have a remainder - r, need hashmap to record r - offset mappings 1/60 = 0.016666
        //r - off - nr - val mapping table
        Map<Long, Integer> hm = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        long nr = 0, v= 0;
        int off = 0; //substring(int) can only take int not long
        String frac = "";
        while (r != 0){
            nr = r * 10;
            v = nr / d;
            sb.append(v);
            hm.put(r, off);
            r = nr % d;

            if (hm.containsKey(r)){
                frac = sb.substring(0, hm.get(r)) + "(" + sb.substring(hm.get(r)) + ")";
                break;
            }
            ++off;
        }
        return (isNeg? "-": "") + integ + "." + (frac.isEmpty()? sb.toString(): frac);
    }

    //167
    public int[] twoSum(int[] numbers, int target) {
        int[] res = {-1, -1};
        if (numbers == null || numbers.length < 2)
            return res;
        int l = 0, r = numbers.length - 1, sum = 0;
        while (l < r){
            sum = numbers[l] + numbers[r];
            if (sum < target)
                ++l;
            else if (sum > target)
                --r;
            else {
                res[0] = l + 1;
                res[1] = r + 1;
                break; //Need break otherwise change pointers
            }
        }
        return res;
    }

    //172
    public int trailingZeroes(int n) {
        int res = 0;
        while (n >= 5){
            int k = n /5;
            res +=k;
            n = k;
        }
        return res;
    }

    //173
    public class BSTIterator {
        Deque<TreeNode> st;

        public BSTIterator(TreeNode root) {
            st = new ArrayDeque<>();
            pushLeft(root);
        }

        private void pushLeft(TreeNode root){
            while (root != null){
                st.push(root);
                root = root.left;
            }
        }

        /** @return whether we have a next smallest number */
        public boolean hasNext() {
            return !st.isEmpty();
        }

        /** @return the next smallest number */
        public int next() {
            TreeNode tn = st.pop();
            pushLeft(tn.right);
            return tn.val;
        }
    }

    //179
    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return "";
        String[] sn = new String[nums.length];
        for (int i = 0; i < nums.length; ++i)
            sn[i] = Integer.toString(nums[i]);
        Arrays.sort(sn, (s1, s2)->(s2 + s1).compareTo(s1 + s2));
        StringBuilder sb = new StringBuilder();
        if (sn[0].equals("0")) //Note all "0" case!!
            return "0";
        for (String s: sn)
            sb.append(s);

        return sb.toString();

    }

    //191
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0){
            n = n & (n - 1);
            ++res;
        }
        return res;
    }

    //199
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int cur = 1, next = 0;
        while (!queue.isEmpty()){
            TreeNode tn = queue.poll();
            if (tn.left != null){
                queue.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                queue.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                res.add(tn.val);
                cur = next;
                next = 0;
            }
        }
        return res;
    }




}
