import java.util.*;

/**
 * Created by zplchn on 12/11/16.
 */
public class Solution300 {
    //303
    public class NumArray1 {
        private int[] dp;

        public NumArray1(int[] nums) {
            if (nums == null || nums.length == 0) //need to have the len = 0 here ; test: null , [], [1], [1,2]
                return;
            dp = new int[nums.length + 1];
            for (int i = 1; i < dp.length; ++i) //to prevent edge cases!! we use a dp with size = n + 1
                dp[i] = nums[i - 1] + dp[i - 1];
        }

        public int sumRange(int i, int j) {
            if (dp == null || i >= dp.length || j >= dp.length || i < 0 || j < 0)
                return 0;
            return dp[j + 1] - dp[i];
        }
    }

    //304
    public class NumMatrix1 {
        private int[][] dp;

        public NumMatrix1(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
                return;
            dp = new int[matrix.length + 1][matrix[0].length + 1];
            for (int i = 0; i < matrix.length; ++i) {
                int sum = 0; //Note: here we sum up the current row plus the block from previous row. NOT plus dp left and dp up. cause it will sum up the blocks over over again
                for (int j = 0; j < matrix[0].length; ++j) {
                    dp[i + 1][j + 1] = matrix[i][j] + dp[i][j + 1] + sum;
                    sum += matrix[i][j];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            if (row1 < 0 || col1 < 0 || row2 < 0 || col2 < 0 || row1 >= dp.length || row2 >= dp.length || col1 >= dp[0].length || col2 >= dp[0].length)
                return 0;
            return dp[row2 + 1][col2 + 1] - dp[row2 + 1][col1] - dp[row1][col2 + 1] + dp[row1][col1]; //See how bump up dp using 1-based helps!!
        }
    }

    //305
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        //Union-Find problem. Union - combine two connected set; Find - check if two obj belong same connected set
        //complxity o(MlogN) M -#of unions N - total # of obj
        List<Integer> res = new ArrayList<>();
        if (m <= 0 || n <= 0 || positions == null || positions.length == 0 || positions[0].length == 0)
            return res;
        //union find needs to create an extra space id[] size = input
        int[] id = new int[m * n];
        Arrays.fill(id, -1); //initially all nodes belong to a dark -1 set - water
        int count = 0;

        for (int[] p : positions) {
            int idx = p[0] * n + p[1];
            id[idx] = idx; //doesnt matter what id we give this time. we will union later
            ++count;

            //now find all 4 neighbours. Note we just need to look the 4 neighbours.
            //NOT DFS. because they all have an asscociate id. we just see if can union - one island
            int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int[] o : off) {
                int x = p[0] + o[0], y = p[1] + o[1], newIdx = x * n + y;
                //still do valid check of indices and if is valid set
                if (x >= 0 && x < m && y >= 0 && y < n && id[newIdx] != -1) { //not water set
                    //union. make their index = mine
                    int root = root(id, newIdx);
                    if (root != idx) { //may have dup input positions
                        id[root] = idx; //union. set as children under idx
                        --count; //every union decrease the count(number of set) by 1
                    }
                }
            }
            res.add(count); //every time log count
        }
        return res;
    }

    private int root(int[] id, int i) { //quick union + path compression
        while (id[i] != i) {
            id[i] = id[id[i]]; //path compression
            i = id[i];
        }
        return i;
    }

    // Segment Tree - Array Implementation - 14ms

// important key is for a range [l,r] of [0, n - 1],
// [l, r] is bound to segment index iSeg = 0;
// [l, mid] is bound to segment index iSeg*2 + 1;
// [mid+1, r] is bound to segment index iSeg*2 + 2;
// whenener [l,r] has l == r, it is a leaf and seg[iSeg] = arr[l];

    class SegmentTreeNode {
        int start, end; //start end are purely for index
        SegmentTreeNode left, right;
        int max; //customized let node storing one kind of data

        SegmentTreeNode(int s, int e) { //Note java does not support default value
            start = s;
            end = e;
        }

        SegmentTreeNode(int s, int e, int max) { //need override
            start = s;
            end = e;
            this.max = max;
        }
    }

    //lint 201 segment tree build, just index
    public SegmentTreeNode build(int start, int end) {
        // write your code here
        if (start > end)
            return null;

        SegmentTreeNode root = new SegmentTreeNode(start, end);
        if (start != end) {
            int m = start + ((end - start) >> 1);
            root.left = build(start, m);
            root.right = build(m + 1, end);
        }
        return root;
    }

    //lint 439 - segment tree build with range max
    public SegmentTreeNode build(int[] A) {
        if (A == null || A.length == 0)
            return null;
        return buildTree(A, 0, A.length - 1);
    }

    private SegmentTreeNode buildTree(int[] A, int s, int e) {
        if (s > e)
            return null;
        SegmentTreeNode root = new SegmentTreeNode(s, e, 0);
        if (s == e)
            root.max = A[s];
        else {
            int m = s + ((e - s) >> 1);
            root.left = buildTree(A, s, m);
            root.right = buildTree(A, m + 1, e);
            root.max = Math.max(root.left.max, root.right.max);
        }
        return root;
    }

    //lint202 - segment tree query
    public int query(SegmentTreeNode root, int start, int end) {
        if (start > end || end < root.start || start > root.end) //outside
            return 0;
        if (start <= root.start && end >= root.end) //all inclusive
            return root.max;
        int m = root.start + ((root.end - root.start) >> 1);
        if (end <= m)
            return query(root.left, start, end);
        else if (start > m)
            return query(root.right, start, end);
        else
            return Math.max(query(root.left, start, m), query(root.right, m + 1, end));
    }

    //lint203 - segment tree modify
    public void modify(SegmentTreeNode root, int index, int value) {
        if (root == null || index < root.start || index > root.end)
            return;
        if (root.start == root.end) {
            root.max = value;
            return;
        }
        int m = root.start + ((root.end - root.start) >> 1);
        if (index > m)
            modify(root.right, index, value);
        else
            modify(root.left, index, value);
        root.max = Math.max(root.left.max, root.right.max);
    }

    //307
    public class NumArray {
        //to achieve o(logn) time, we build a segment tree
        class SegmentTreeNode {
            int start, end, sum;
            SegmentTreeNode left, right;

            SegmentTreeNode(int s, int e, int sum) {
                start = s;
                end = e;
                this.sum = sum;
            }
        }

        private SegmentTreeNode root;
        private int[] nums;

        public NumArray(int[] nums) {
            if (nums == null || nums.length == 0)
                return;
            root = buildTree(nums, 0, nums.length - 1);
            this.nums = nums;
        }

        void update(int i, int val) {
            if (i < 0 || i >= nums.length)
                return;
            int diff = val - nums[i];
            nums[i] = val; //note here must upate in the array as well!!!!
            updateTree(root, i, diff); //give the delta
        }

        public int sumRange(int i, int j) {
            return queryTree(root, i, j);
        }

        private void updateTree(SegmentTreeNode root, int i, int delta) {
            if (root == null || i < root.start || i > root.end)
                return;
            int m = root.start + ((root.end - root.start) >> 1);
            if (root.start != root.end) {
                if (i > m)
                    updateTree(root.right, i, delta);
                else
                    updateTree(root.left, i, delta);
            }
            root.sum += delta;
        }

        private int queryTree(SegmentTreeNode root, int i, int j) {
            if (root == null || i > j || j < root.start || i > root.end)
                return 0;
            if (i <= root.start && j >= root.end)
                return root.sum;
            int m = root.start + ((root.end - root.start) >> 1);
            if (i > m)
                return queryTree(root.right, i, j);
            if (j <= m)
                return queryTree(root.left, i, j);
            return queryTree(root.left, i, m) + queryTree(root.right, m + 1, j);
        }

        private SegmentTreeNode buildTree(int[] nums, int l, int r) {
            if (l > r)
                return null;
            int m = l + ((r - l) >> 1);
            SegmentTreeNode root = new SegmentTreeNode(l, r, 0);
            if (l == r)
                root.sum = nums[l];
            else {
                root.left = buildTree(nums, l, m);
                root.right = buildTree(nums, m + 1, r);
                root.sum = root.left.sum + root.right.sum;
            }
            return root;
        }
    }

    //308
    public class NumMatrix {
        private int[][] dp;

        public NumMatrix(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
                return;
            dp = new int[matrix.length + 1][matrix[0].length + 1];
            for (int i = 0; i < matrix.length; ++i) {
                for (int j = 0; j < matrix[0].length; ++j) {
                    dp[i + 1][j + 1] = matrix[i][j] + dp[i + 1][j]; //only do sum of the cur row to the left
                }
            }
        }

        public void update(int row, int col, int val) {
            if (row < 0 || row >= dp.length - 1 || col < 0 || col >= dp[0].length - 1)
                return;
            int diff = val - (dp[row + 1][col + 1] - dp[row + 1][col]);
            for (int j = col + 1; j < dp[0].length; ++j)
                dp[row + 1][j] += diff;
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            if (row1 < 0 || col1 < 0 || row2 < 0 || col2 < 0 || row1 >= dp.length - 1 || row2 >= dp.length - 1 || col1 >= dp[0].length - 1 || col2 >= dp[0].length - 1)
                return 0;
            int res = 0;
            for (int i = row1 + 1; i <= row2 + 1; ++i)
                res += dp[i][col2 + 1] - dp[i][col1];
            return res;
        }
    }

    //314
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Map<Integer, List<Integer>> tm = new TreeMap<>();
        Queue<TreeNode> qt = new LinkedList<>(); //Simultaneous two maps record same thing in parallel
        Queue<Integer> qi = new LinkedList<>();
        qt.offer(root);
        qi.offer(0);

        while (!qt.isEmpty()) {
            TreeNode tn = qt.poll();
            int to = qi.poll();
            tm.putIfAbsent(to, new ArrayList<Integer>());
            tm.get(to).add(tn.val);

            if (tn.left != null) {
                qt.offer(tn.left);
                qi.offer(to - 1);
            }
            if (tn.right != null) {
                qt.offer(tn.right);
                qi.offer(to + 1);
            }
        }
        res.addAll(tm.values()); //TreeMap implements SortedMap<K,V> and the latter has a values() methods defined returning ascending ordered key view - values
        return res;
    }

//    private void verticalHelper(TreeNode root, int off, Map<Integer, List<Integer>> tm){ //DFS is wrong. a left tree child can go all the way its right child and offset into right tree
    // and in that case, the deep left tree child will be added ahead of a right tree early child
//        if (root == null)
//            return;
//        tm.putIfAbsent(off, new ArrayList<Integer>());
//        tm.get(off).add(root.val);
//        verticalHelper(root.left, off - 1, tm);
//        verticalHelper(root.right, off + 1, tm);
//    }

    //315
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        //create a bst with node contains count of left children and count of self dup
        BSTNode root = null;
        //insert node on to the bst from right to left since we care smaller on the right side
        Integer[] tres = new Integer[nums.length];
        for (int i = nums.length - 1; i >= 0; --i) {
            root = insertBST(root, nums, i, tres, 0);
        }
        return Arrays.asList(tres); //tres must use Integer[] inorder to convert from array to List
    }

    class BSTNode {
        int val, leftCnt, dup = 1; //note dup by default is 1
        BSTNode left, right;

        BSTNode(int x) {
            val = x;
        }
    }

    private BSTNode insertBST(BSTNode root, int[] nums, int i, Integer[] res, int preSum) {
        if (root == null) {
            root = new BSTNode(nums[i]);
            res[i] = preSum;
        } else if (root.val == nums[i]) {
            ++root.dup;
            res[i] = preSum + root.leftCnt; //a dup may have left children already when the dup comes in. so stop and directly add leftCnt 2,1,2 right to left
        } else if (root.val < nums[i]) {
            root.right = insertBST(root.right, nums, i, res, preSum + root.leftCnt + root.dup); //when go right, add preSum + everything at cur node
        } else {
            ++root.leftCnt; //bump up leftCnt for left children
            root.left = insertBST(root.left, nums, i, res, preSum); //when go left, cannot add just carry over preSum
        }
        return root;
    }

    //316
    public String removeDuplicateLetters(String s) {
        //count the freq of s. then loop through char by char. and maintain a deque storing looped char.
        //if a char is less than the last stacked char and that last char still has more in the freq table, pop it out.
        //use a visited table to store good chars since the stack will manage a roughly asceding ones(unless for those dont have enough dups)
        //so we need to ignore the same char if comes in again but already set in the deque ; bcabd the second b comes and cannot do anything
        //when a char is kicked off by a smaller char, mark the visited to taht char to false again.
        if (s == null || s.length() == 0)
            return s;
        //first count freq of chars;
        int[] freq = new int[256];
        for (int i = 0; i < s.length(); ++i)
            ++freq[s.charAt(i)];
        boolean[] visited = new boolean[256]; // "ccccc"
        Deque<Character> dq = new ArrayDeque<>();
        for (int i = 0; i < s.length(); ++i) {
            --freq[s.charAt(i)];
            if (visited[s.charAt(i)])
                continue;
            while (!dq.isEmpty() && s.charAt(i) < dq.peek() && freq[dq.peek()] > 0)
                visited[dq.pop()] = false; //treat u as not visited. still have chance
            dq.push(s.charAt(i));
            visited[s.charAt(i)] = true;
        }
        StringBuilder sb = new StringBuilder();
        while (!dq.isEmpty())
            sb.append(dq.pollLast());
        return sb.toString();
    }

    //317
    public int shortestDistance(int[][] grid) {
        // start from every single "1", do BFS, need mark visited, use neg, lvl, if 2 not adding.
        // at the end, loop again and find the max neg. is the total

        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int[][] reached_houses = new int[grid.length][grid[0].length];
        int total_houses = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] == 1) {
                    ++total_houses;
                    shortestDistanceHelper(grid, i, j, reached_houses, new boolean[grid.length][grid[0].length]);
                }
            }
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (reached_houses[i][j] == total_houses && grid[i][j] < 0)
                    res = Math.max(res, grid[i][j]);
            }
        }
        return res == Integer.MIN_VALUE ? -1 : -res;
    }

    private final int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private void shortestDistanceHelper(int[][] grid, int i, int j, int[][] reached_houses, boolean[][] visited) {
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{i, j}); //NOTE, use this way cannot have 2 in the []!!!!!! new int[]{1,2} create and init cannot have size
        visited[i][j] = true;
        int cur = 1, next = 0, lvl = -1;
        while (!queue.isEmpty()) {
            int[] t = queue.poll();
            for (int k = 0; k < off.length; ++k) {
                int nx = t[0] + off[k][0], ny = t[1] + off[k][1];
                if (nx >= 0 && nx < grid.length && ny >= 0 && ny < grid[0].length && !visited[nx][ny] && grid[nx][ny] <= 0) {
                    queue.offer(new int[]{nx, ny});
                    ++next;
                    visited[nx][ny] = true;
                    reached_houses[nx][ny] += 1;
                    grid[nx][ny] += lvl;
                }
            }
            if (--cur == 0) {
                cur = next;
                next = 0;
                lvl -= 1;
            }
        }
    }

    //318
    public int maxProduct(String[] words) {
        if (words == null || words.length == 0)
            return 0;
//        BitSet[] bset = new BitSet[words.length];
//        for (int i = 0; i < words.length; ++i){
//            BitSet bs = new BitSet();
//            for (int j = 0; j < words[i].length(); ++j)
//                bs.set(words[i].charAt(j) - 'a');
//            bset[i] = bs;
//        }
        int[] bset = new int[words.length]; //much faster than BitSet. so for boolean checklist of chars set in a string, use a simple int as bitmap because 32 is enough to handle, 1<<offset
        for (int i = 0; i < words.length; ++i) {
            for (int j = 0; j < words[i].length(); ++j) {
                bset[i] |= (1 << (words[i].charAt(j) - 'a')); //use a 32bit int as a bitmap
            }
        }
        int res = 0;
        for (int i = 1; i < words.length; ++i) {
            int maxj = 0;
            for (int j = 0; j < i; ++j) {
                if ((bset[i] & bset[j]) == 0)
                    maxj = Math.max(maxj, words[j].length());
            }
            res = Math.max(res, words[i].length() * maxj);
        }
        return res;
    }

    //320
    public List<String> generateAbbreviations(String word) {
        List<String> res = new ArrayList<>();
        if (word == null)
            return res;
        abbrHelper(word, 0, "", res);
        return res;
    }

    private void abbrHelper(String word, int start, String pre, List<String> res) { //point is change or Not change. when change, if the previous is already a digit, combine !!! so no consecutive digits
        if (start == word.length()) {
            res.add(pre);
            return;
        }
        abbrHelper(word, start + 1, pre + word.charAt(start), res);
        if (start > 0 && Character.isDigit(pre.charAt(pre.length() - 1))) {
            int x = pre.charAt(pre.length() - 1) - '0' + 1; //or Character.getNumericValue(ch) + 1; '0'-'9'a -> 0 -9; 'a'/'A' - 10, 'j'/'J' -> 16. this is NOT ascii value but just a special mapping ignoring case
            String t = pre.substring(0, pre.length() - 1) + x; // string + 1 + 2 --> string12. every partion of + will be treat as string,
            abbrHelper(word, start + 1, t, res);
        } else
            abbrHelper(word, start + 1, pre + 1, res);
    }

    //322
    public int coinChange(int[] coins, int amount) { //Note coin change is NOT a greedy problem!!! it's a DP problem!!!!
        if (coins == null || coins.length == 0 || amount <= 0)
            return 0;
        Arrays.sort(coins);
        //dp[i] = min(dp[coin[j]] + dp[i-coin[j]])
        int[] dp = new int[amount + 1]; //use amount + 1 not INF so the dp+ wont overflow
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; ++i) {
            for (int j = 0; j < coins.length && i >= coins[j]; ++j) {
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1); //only related to last time use a j and the rest
            }
        }
        return dp[dp.length - 1] > amount ? -1 : dp[dp.length - 1];
    }

    //323
    public int countComponents(int n, int[][] edges) {
        if (n <= 1 || edges == null || edges.length == 0 || edges[0].length == 0)
            return n;
        //start from any unmasked node and do dfs on graph and mask.
        //first given edges --> convert to adjacency list
        List<Integer>[] children = new List[n];
        for (int i = 0; i < children.length; ++i)
            children[i] = new ArrayList<>();
        for (int[] e : edges) {
            children[e[0]].add(e[1]); //Graph search, need to set up adj lists. since we do flood fill, doesnt matter which one to start. so no need to have indegree list.
            children[e[1]].add(e[0]);
        }
        boolean[] visited = new boolean[n];
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                ++res;
                countHelper(visited, i, children);
            }
        }
        return res;
    }

    private void countHelper(boolean[] visited, int i, List<Integer>[] children) { //need to pass down the children so that children's children can be fetched.
        visited[i] = true;
        for (int c : children[i]) {
            if (!visited[c])
                countHelper(visited, c, children);
        }
    }

    //337
    public int rob(TreeNode root) {
        if (root == null)
            return 0;
        int[] res = robHelper(root);
        return Math.max(res[0], res[1]);
    }

    private int[] robHelper(TreeNode root) {
        int[] res = new int[2];
        if (root == null)
            return res;
        int[] lres = robHelper(root.left);
        int[] rres = robHelper(root.right);
        res[0] = Math.max(lres[1], lres[0]) + Math.max(rres[0], rres[1]); //note here is find the max of the two children, need to test both include/exclude cases
        res[1] = root.val + lres[0] + rres[0];
        return res;
    }

    //347
    public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0 || k <= 0)
            return res;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int x : nums) {
            hm.put(x, hm.getOrDefault(x, 0) + 1);
        }
        // Find top k method 1 - o(nlogk) pq
        Queue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>((e1, e2) -> e1.getValue() - e2.getValue());
        for (Map.Entry<Integer, Integer> e : hm.entrySet()) {
            if (pq.size() < k)
                pq.offer(e);
            else if (e.getValue() > pq.peek().getValue()) {
                pq.poll();
                pq.offer(e);
            }
        }
        for (Map.Entry<Integer, Integer> e : pq)
            res.add(e.getKey());

        //Find top k method 2 bucket sort - o(n) k is limited under n
        List<Integer>[] bucket = new List[nums.length + 1];
        for (Map.Entry<Integer, Integer> e : hm.entrySet()) {
            if (bucket[e.getValue()] == null)
                bucket[e.getValue()] = new ArrayList<>();
            bucket[e.getValue()].add(e.getKey());
        }
        for (int i = bucket.length - 1; i >= 0; --i) {
            if (bucket[i] != null) {
                for (int x : bucket[i]) {
                    res.add(x);
                    if (res.size() == k)
                        return res;
                }
            }
        }

        return res;
    }

    //342
    public boolean isPowerOfFour(int num) {
        return (num > 0) && (num & 0x55555555) == num && (num & (num - 1)) == 0;     //A int only has one 1 bit -> n&n-1 == 0 ; An int is 0xFFFFFFFF 8 bytes len. 0101 0101 is 4 power
    }

    //349
    public int[] intersection(int[] nums1, int[] nums2) {
        if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0)
            return new int[0];
        if (nums1.length > nums2.length)
            return intersection(nums2, nums1);
        Set<Integer> hs = new HashSet<>();
        for (int x : nums1) {
            hs.add(x);
        }
        List<Integer> list = new ArrayList<>();
        for (int y : nums2) {
            if (hs.contains(y)) {
                list.add(y);
                hs.remove(y); //Result does not allow dup.
            }
        }
        int[] res = new int[list.size()];
        int i = 0;
        for (int x : list)
            res[i++] = x; //Note: There is no way to directly convert Collecions/Arrays to primitive int[]. For Integer[] can use list.toArray(new Integer[list.size()]);
        return res;
    }

    //358
    public String rearrangeString(String str, int k) {
        if (str == null || str.length() == 0 || k <= 1)
            return str;
        //1. count char by freq and enter into a map
        Map<Character, Integer> hm = new HashMap<>();
        for (int i = 0; i < str.length(); ++i) {
            hm.put(str.charAt(i), hm.getOrDefault(str.charAt(i), 0) + 1);
        }
        //2. we set up a pq which sort by hm's value big first
        Queue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((e1, e2) -> e2.getValue() - e1.getValue());
        //3. we set up a treemap<int, entry> which is when at index i, the first entry is okay to be put back in pq
        TreeMap<Integer, Map.Entry<Character, Integer>> tm = new TreeMap<>();
        StringBuilder sb = new StringBuilder();
        pq.addAll(hm.entrySet()); //dont forget to put in
        for (int i = 0; i < str.length(); ++i) {
            //first put back any on the tm to pq
            if (!tm.isEmpty() && tm.firstKey() == i)
                pq.offer(tm.pollFirstEntry().getValue());
            if (pq.isEmpty())
                return "";
            Map.Entry<Character, Integer> e = pq.poll();
            sb.append(e.getKey());
            if (e.getValue() > 1) {
                e.setValue(e.getValue() - 1); //Map.Entry<k,v> getValue(), setValue()
                tm.put(i + k, e);
            }
        }
        return sb.toString();
    }

    //360
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        //parabola's middle axis -b/2a. so we use two pointers, if open up, we will fill from biggest to least. if open down, we will fill from smallest to biggest
        if (nums == null || nums.length == 0)
            return nums;
        int[] res = new int[nums.length]; //when return a int[], put this after the check when return itself to avoid input is a null
        int l = 0, r = nums.length - 1, idx = a > 0 ? res.length - 1 : 0;
        while (l <= r) { //Note here need =. the last one or only one still needs to be filled!
            int lr = transHelper(a, b, c, nums[l]);
            int rr = transHelper(a, b, c, nums[r]);
            if (a > 0) {
                if (lr < rr) {
                    res[idx--] = rr;
                    --r;
                } else {
                    res[idx--] = lr;
                    ++l;
                }
            } else {
                if (lr < rr) {
                    res[idx++] = lr;
                    ++l;
                } else {
                    res[idx++] = rr;
                    --r;
                }
            }
        }
        return res;
    }

    private int transHelper(int a, int b, int c, int x) {
        return a * x * x + b * x + c;
    }

    //361
    public int maxKilledEnemies(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        //think about first in 1-d array. we send a bomb and the E we get is the same for all empty spot till the first W. so check once is enough. when pass another Wall, check again
        //and for 2d. we apply the same thing to column. only at 0 or just pass a wall we check(even we are at a E or W). and we check the max by checking if we are at a empty spot
        int rowEnemies = 0, res = 0;
        int[] colEnemies = new int[grid[0].length];
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (j == 0 || grid[i][j - 1] == 'W') {
                    rowEnemies = 0; //This should reset to 0 at here!!!
                    for (int k = j; k < grid[0].length && grid[i][k] != 'W'; ++k)
                        rowEnemies += grid[i][k] == 'E' ? 1 : 0;
                }
                if (i == 0 || grid[i - 1][j] == 'W') {
                    colEnemies[j] = 0; //reset to 0!!
                    for (int k = i; k < grid.length && grid[k][j] != 'W'; ++k)
                        colEnemies[j] += grid[k][j] == 'E' ? 1 : 0;
                }
                if (grid[i][j] == '0') //we only check when at an empty spot
                    res = Math.max(res, rowEnemies + colEnemies[j]);
            }
        }
        return res;
    }

    //362
    public class HitCounter { //check Map<k,v> shm = Collections.synchronizedMap(new HashMap<k,v>()); for multithreaded access at the same time
        private TreeMap<Integer, Integer> tm;
        private int sum;

        /**
         * Initialize your data structure here.
         */
        public HitCounter() {
            tm = new TreeMap<>();
        }

        /**
         * Record a hit.
         *
         * @param timestamp - The current timestamp (in seconds granularity).
         */
        public void hit(int timestamp) {
            tm.put(timestamp, tm.getOrDefault(timestamp, 0) + 1);
            ++sum;
            //adjust(timestamp); //this could be discussed if needed.  if do not do this step here it's much faster.
        }

        /**
         * Return the number of hits in the past 5 minutes.
         *
         * @param timestamp - The current timestamp (in seconds granularity).
         */
        public int getHits(int timestamp) { //here it asks for at any given time, what is the total. not the total since last time hits!!!. so here also need to remove obsolete rec
            adjust(timestamp);
            return sum;
        }

        private void adjust(int timestamp) {
            //here the ts maybe empty
            while (!tm.isEmpty() && tm.firstKey() <= timestamp - 300)
                sum -= tm.pollFirstEntry().getValue();
        }
    }

    //366
    public List<List<Integer>> findLeaves(TreeNode root) { //at same height from the ground up will be added at the same time
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        //the height we meet is continously increasing, start from 0, so can directly using index of the res
        findLeavesHelper(root, res);
        return res;
    }

    private int findLeavesHelper(TreeNode root, List<List<Integer>> res) {
        if (root == null)
            return 0;
        int lh = findLeavesHelper(root.left, res);
        int rh = findLeavesHelper(root.right, res);
        int h = Math.max(lh, rh) + 1;
        if (res.size() < h)
            res.add(new ArrayList<>());
        res.get(h - 1).add(root.val);
        return h;
    }

    //368
    public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        Arrays.sort(nums); //this is must!!!
        int[] dp = new int[nums.length];
        int[] lastStep = new int[nums.length]; //best example for showing how to use DP and output the trace as well!!
        Arrays.fill(lastStep, -1); // 0 is a valid index for a step. so use -1
        Arrays.fill(dp, 1); //any given num at least itself a subset
        int maxDp = 1, maxIdx = 0;
        for (int i = 1; i < dp.length; ++i) {
            for (int j = i - 1; j >= 0; --j) { //3, 6, 8, 24. must look till 0
                if (nums[i] % nums[j] == 0) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        lastStep[i] = j;
                    }
                    if (dp[i] > maxDp) {
                        maxDp = dp[i];
                        maxIdx = i;
                    }
                }
            }
        }
        for (int i = maxIdx; i >= 0; i = lastStep[i])
            res.add(nums[i]);
        return res;
    }

    //369
    public ListNode plusOne(ListNode head) {
        if (head == null)
            return head;
        ListNode cur = head, lastN9 = null;
        while (cur != null) { //trick is here we do not need to reverse the whole list back and forth. we use a last Non 9 guard varible to log where the last non9 and set all 0 after it
            if (cur.val != 9)
                lastN9 = cur;
            cur = cur.next;
        }
        if (lastN9 == null) {
            lastN9 = new ListNode(1);
            lastN9.next = head;
            head = lastN9;
        } else
            ++lastN9.val;
        cur = lastN9.next;
        while (cur != null) {
            cur.val = 0;
            cur = cur.next;
        }
        return head;
    }

    //370
    public int[] getModifiedArray(int length, int[][] updates) {
        //we only care about the border where change happens, where entering, we bump, where gets out we decrease at the next slot. then we basically add to the previos by one more pass
        if (length <= 0 || updates == null)
            return new int[0];
        int[] res = new int[length];
        for (int[] u : updates) {
            res[u[0]] += u[2];
            if (u[1] + 1 < res.length) //note check range
                res[u[1] + 1] -= u[2];
        }
        for (int i = 1; i < res.length; ++i)
            res[i] += res[i - 1];
        return res;
    }

    //379
    public class PhoneDirectory {
        private BitSet bs;
        private int size, cap;

        /**
         * Initialize your data structure here
         *
         * @param maxNumbers - The maximum numbers that can be stored in the phone directory.
         */
        public PhoneDirectory(int maxNumbers) {
            bs = new BitSet();
            cap = maxNumbers;
        }

        /**
         * Provide a number which is not assigned to anyone.
         *
         * @return - Return an available number. Return -1 if none is available.
         */
        public int get() {
            if (size == cap)
                return -1;
            int x = bs.nextClearBit(0); //BitSet.nextClearBit(int from); previosClearBit(int from), nextSetBit(int from)
            bs.set(x);
            ++size;
            return x;
        }

        /**
         * Check if a number is available or not.
         */
        public boolean check(int number) {
            if (number >= cap)
                return false;
            return bs.get(number) == false;
        }

        /**
         * Recycle or release a number.
         */
        public void release(int number) {
            if (number < cap) {
                if (bs.get(number)) { //Here need to first check if this is a set number!!!
                    bs.set(number, false);
                    --size; //because size needs to be adjusted
                }
            }
        }
    }

    //380
    public class RandomizedSet {
        Map<Integer, Integer> hm;
        List<Integer> list;

        /**
         * Initialize your data structure here.
         */
        public RandomizedSet() {
            hm = new HashMap<>();
            list = new ArrayList<>();
        }

        /**
         * Inserts a value to the set. Returns true if the set did not already contain the specified element.
         */
        public boolean insert(int val) {
            if (hm.containsKey(val))
                return false;
            list.add(val);
            hm.put(val, list.size() - 1);
            return true;
        }

        /**
         * Removes a value from the set. Returns true if the set contained the specified element.
         */
        public boolean remove(int val) {
            if (!hm.containsKey(val))
                return false;
            int idx = hm.get(val);
            hm.remove(val);
            if (idx != list.size() - 1) { //when last one, no rearrange ,must delete!
                int t = list.get(list.size() - 1);
                hm.put(t, idx);
                list.set(idx, t);
            }
            list.remove(list.size() - 1);
            return true;
        }

        /**
         * Get a random element from the set.
         */
        public int getRandom() {
            return list.get(new Random().nextInt(list.size()));
        }
    }

    //381
    public class RandomizedCollection {
        private List<Integer> list;
        private Map<Integer, Set<Integer>> hm;
        private Random random;

        /**
         * Initialize your data structure here.
         */
        public RandomizedCollection() {
            list = new ArrayList<>();
            hm = new HashMap<>();
            random = new Random();
        }

        /**
         * Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
         */
        public boolean insert(int val) {
            boolean res = true;
            if (hm.containsKey(val))
                res = false;
            else
                hm.put(val, new HashSet<>());
            list.add(val);
            hm.get(val).add(list.size() - 1);
            return res;
        }

        /**
         * Removes a value from the collection. Returns true if the collection contained the specified element.
         */
        public boolean remove(int val) {
            if (!hm.containsKey(val))
                return false; //first need to check if the val is a valid one
            int lastIdx = list.size() - 1;
            int last = list.get(lastIdx);
            list.remove(lastIdx);
            hm.get(last).remove(lastIdx);
            //note if the one is the last one in the list
            if (last == val) {//if the one is the last one we dont need to swap
                if (hm.get(last).isEmpty()) //if not the last one we will add one to this set anyways so dont delete
                    hm.remove(last);
                return true;
            }
            Iterator<Integer> iter = hm.get(val).iterator();
            int valIdx = iter.next();
            iter.remove(); //Iterator.remove() removes the last item returned by a next()
            if (hm.get(val).isEmpty())
                hm.remove(val);
            list.set(valIdx, last);
            hm.get(last).add(valIdx);
            return true;
        }

        /**
         * Get a random element from the collection.
         */
        public int getRandom() {
            if (list.isEmpty())
                return -1;
            return list.get(random.nextInt(list.size()));
        }
    }

    //382
    public class Solution1 {
        private ListNode head; //use private!!
        private Random random; //we dont want to create a random obj every time

        /**
         * @param head The linked list's head.
         *             Note that the head is guaranteed to be not null, so it contains at least one node.
         */
        public Solution1(ListNode head) {
            this.head = head;
            this.random = new Random();
        }

        /**
         * Returns a random node's value.
         */
        public int getRandom() {
            int res = head.val; //set the one in resevoir k =1
            ListNode cur = head.next;
            int len = 1;
            while (cur != null) {
                ++len;
                if (random.nextInt(len) == 0)
                    res = cur.val;
                cur = cur.next; //dont forget to advance!!!
            }
            return res;
        }
    }

    //384
    public class Solution3 {
        private int[] onums;
        private int[] tnums;
        private Random random;

        public Solution3(int[] nums) {
            this.onums = nums;
            random = new Random();
            reset();
        }

        /**
         * Resets the array to its original configuration and return it.
         */
        public int[] reset() {
            tnums = Arrays.copyOf(onums, onums.length); //int[] copy = Arrays.copyco(int[]x, len);
            return tnums;
        }

        /**
         * Returns a random shuffling of the array.
         */
        public int[] shuffle() {
            //shuffle algorithm: everytime get a random number up to CURRENT position, swap cur elem to the random. NOTE [1,2] MUST have a chance to stay. not always go to 2,1
            for (int i = 0; i < tnums.length; ++i) {
                int rdm = random.nextInt(i + 1); //include current pos
                int t = tnums[i];
                tnums[i] = tnums[rdm];
                tnums[rdm] = t;
            }
            return tnums;
        }
    }

    //388
    public int lengthLongestPath2(String input) {
        //use stack to store the length of sum of length of dir, stack size will be the current depth. when meet a file, get the count and update max
        if (input == null || input.length() == 0)
            return 0;
        String[] tokens = input.split("\\n");
        Deque<Integer> st = new ArrayDeque<>();
        int max = 0;

        for (String s : tokens) {
            int depth = s.lastIndexOf('\t') + 1; //note \t or \n is a ascii char !!! string.lastIndexOf(char/string) return the start index from the right, -1 if not found
            //adjust level
            while (depth < st.size())
                st.pop();
            int len = s.substring(depth).length();

            if (s.indexOf('.') != -1)//it's a file
                max = Math.max(max, len + (st.isEmpty() ? 0 : st.peek() + st.size()));
            else {//it's a dir
                st.push(len + (st.isEmpty() ? 0 : st.peek()));
            }
        }
        return max;

    }

    public int lengthLongestPath(String input) { //this is the one without extra space
        if (input == null || input.length() == 0)
            return 0;
        int i = -1, max = 0; //here the trick is to set the start i to -1. so search start at i + 1.
        Deque<Integer> st = new ArrayDeque<>();

        while (i < input.length()) {
            //first find next \n, then backwards find last \t. check the #of \t and adjsut stack
            //if it's a file. update len
            int idn = input.indexOf('\n', i + 1); // need to start at i+1 not itslef dead loop
            idn = idn < 0 ? input.length() : idn;
            int idt = input.lastIndexOf('\t', idn);
            idt = idt <= i ? i : idt;
            int lvl = idt - i;
            while (st.size() > lvl)
                st.pop();
            String str = input.substring(idt + 1, idn);
            if (str.indexOf('.') < 0)
                st.push(st.isEmpty() ? str.length() : st.peek() + str.length());
            else {
                int len = st.isEmpty() ? str.length() : st.peek() + st.size() + str.length();
                max = Math.max(max, len);
            }
            i = idn;
        }
        return max;
    }

    //389
    public char findTheDifference(String s, String t) {
        if (s == null || t == null || s.length() + 1 != t.length())
            return 0;
        char res = 0;
        for (int i = 0; i < s.length(); ++i)
            res ^= s.charAt(i);
        for (int i = 0; i < t.length(); ++i)
            res ^= t.charAt(i);
        return res;
    }

    //391
    public boolean isRectangleCover(int[][] rectangles) {
        if (rectangles == null || rectangles.length == 0 || rectangles[0].length != 4)
            return false;
        /* Our thoughts is each rec provide 4 vertex. all middle points being it's two rec back-by-back or 4 together
        they will all cancel out. Only there should be 4 vertexes left.  --> criteria 1
        If there are two separate rec, there will be > 4 left
        Second observation is if one is overlapped by half of the original. it still cancel 2 but provide 2 new
        still end with 4 --> so we will also need to check the sum area of all == final one  --> criteria 2
         */
        //we need to log the final outer 4 in order to check area at last!
        int minx, miny, maxx, maxy, sum = 0; //sum of all area
        minx = miny = Integer.MAX_VALUE;
        maxx = maxy = Integer.MIN_VALUE;
        //Set<int[]> NOTE: array1.equals(array2) is same as arr1 == arr2. not compare content!!!
        //so use string concat the x, y coordinates
        Set<String> hs = new HashSet<>();
        for (int[] r : rectangles) {

            minx = Math.min(minx, r[0]);
            miny = Math.min(miny, r[1]);
            maxx = Math.max(maxx, r[2]);
            maxy = Math.max(maxy, r[3]);

            sum += (r[2] - r[0]) * (r[3] - r[1]);

            String[] vertex = {r[0] + "," + r[1], r[0] + "," + r[3], r[2] + "," + r[1], r[2] + "," + r[3]};
            for (String v : vertex) {
                if (hs.contains(v))
                    hs.remove(v);
                else
                    hs.add(v);
            }
        }
        //Note here the min/max vertexes may already be cancelled out and the set may still end with 4. area may still == obsolete minx/ y
        if (hs.size() != 4 || !hs.contains(minx + "," + miny) || !hs.contains(minx + "," + maxy) || !hs.contains(maxx + "," + miny) || !hs.contains(maxx + "," + maxy))
            return false;
        int bigArea = (maxx - minx) * (maxy - miny); //if outer is okay, but heart repeat twice. points calcelled but area contribute
        return bigArea == sum;
    }


    public boolean isRectangleCover2(int[][] rectangles) {
        if (rectangles == null || rectangles.length == 0) return false;
        int x1 = Integer.MAX_VALUE, x2 = Integer.MIN_VALUE,
                y1 = Integer.MAX_VALUE, y2 = Integer.MIN_VALUE;
        Set<String> set = new HashSet<>();
        int area = 0;
        for (int[] rect : rectangles) {
            x1 = Math.min(x1, rect[0]);
            y1 = Math.min(y1, rect[1]);
            x2 = Math.max(x2, rect[2]);
            y2 = Math.max(y2, rect[3]);

            area += (rect[2] - rect[0]) * (rect[3] - rect[1]);

            String s1 = rect[0] + " " + rect[1];
            String s2 = rect[0] + " " + rect[3];
            String s3 = rect[2] + " " + rect[1];
            String s4 = rect[2] + " " + rect[3];

            if (set.contains(s1))
                set.remove(s1);
            else
                set.add(s1);

            if (set.contains(s2))
                set.remove(s2);
            else
                set.add(s2);

            if (set.contains(s3))
                set.remove(s3);
            else
                set.add(s3);

            if (set.contains(s4))
                set.remove(s4);
            else
                set.add(s4);
        }
        if (!set.contains(x1 + " " + y1) || !set.contains(x1 + " " + y2) || !set.contains(x2 + " " + y1) || !set.contains(x2 + " " + y2) || set.size() != 4)
            return false;
        return area == (x2 - x1) * (y2 - y1);
    }

    //394
    public String decodeString(String s) {
        if (s == null || s.length() == 0)
            return s;
        Deque<StringBuilder> sbq = new ArrayDeque<>();
        Deque<Integer> numq = new ArrayDeque<>();
        int cnt = 0;
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < s.length(); ++i){
            char c = s.charAt(i);
            if (Character.isDigit(c))
                cnt = cnt * 10 + c - '0';
            else if (c == '['){
                numq.push(cnt);
                cnt = 0;
                sbq.push(sb);
                sb = new StringBuilder();
            }
            else if (c == ']'){
                int rep = numq.pop();
                StringBuilder prevSb = sbq.pop();
                for (int j = 0; j < rep; ++j)
                    prevSb.append(sb);
                sb = prevSb;
            }
            else
                sb.append(c);
        }
        return sb.toString();
    }

    //399
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        //directed weighted graph, total # of nodes unknown. use two hashmap to maintain the graph in parallel.
        //This q is find a path between two nodes on a directed graph and multiply all weight along the path is the result

        //construct adjacency list using hm
        Map<String, List<String>> children = new HashMap<>();
        Map<String, List<Double>> weights = new HashMap<>();

        for (int i = 0; i < equations.length; ++i){
            String[] e = equations[i];
            children.putIfAbsent(e[0], new ArrayList<>());
            children.putIfAbsent(e[1], new ArrayList<>());
            weights.putIfAbsent(e[0], new ArrayList<>());
            weights.putIfAbsent(e[1], new ArrayList<>());

            children.get(e[0]).add(e[1]);
            children.get(e[1]).add(e[0]);
            weights.get(e[0]).add(values[i]);
            weights.get(e[1]).add(1 / values[i]);
        }

        //need to find path between two side on the queries
        double[] res = new double[queries.length];
        for (int i = 0; i < queries.length; ++i){
            if (!children.containsKey(queries[i][0]) || !children.containsKey(queries[i][1]))
                res[i] = -1.0;
            else {
                Double d = findPath(queries[i][0], queries[i][1], children, weights, new HashSet<String>(), 1);
                res[i] = d == null? -1.0: d;
            }
        }
        return res;
    }

    private Double findPath(String start, String end, Map<String, List<String>> children, Map<String, List<Double>> weights, Set<String> visited, double mul){
        if (start.equals(end)) {//found
            return mul;
        }
        if (visited.contains(start)) //dont forget to check visited!!!
            return null;
        Double res = null;
        List<String> childNodes = children.get(start);
        List<Double> childWeights = weights.get(start);
        visited.add(start);
        for (int i = 0; i < childNodes.size(); ++i){
            if ((res = findPath(childNodes.get(i), end, children, weights, visited, mul * childWeights.get(i))) != null)
                break;
        }
        return res;
    }


}