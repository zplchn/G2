import java.util.*;

/**
 * Created by zplchn on 12/11/16.
 */
public class Solution200 {
    //200
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == '1'){
                    ++res;
                    numIslandsHelper(grid, i, j);
                }
            }
        }
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if ((grid[i][j] & 256) != 0)
                    grid[i][j] ^= 256;
            }
        }
        return res;
    }
    private final int[][] noff = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    private void numIslandsHelper(char[][] grid, int i, int j){
        Queue<int[]> queue = new LinkedList<>();
        grid[i][j] ^= 256;
        queue.offer(new int[]{i, j});
        while (!queue.isEmpty()) {
            int[] t = queue.poll();
            for (int k = 0; k < noff.length; ++k) {
                int x = t[0] + noff[k][0], y = t[1] + noff[k][1];
                if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && (grid[x][y] & 256) == 0 && grid[x][y] == '1') {
                    grid[x][y] ^= 256;
                    queue.offer(new int[]{x, y});
                }
            }
        }
    }

    //212
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        if (board == null || words == null || words.length == 0)
            return res;
        TrieNode root = constructTrie(words);
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                findHelper(board, root, i, j, new StringBuilder(), res);   //start from every single spot!!!
            }
        }

        Set<String> hs = new HashSet(res); //Need to remove dup!!!
        res = new ArrayList<>();
        res.addAll(hs);
        return res;
    }

    class TrieNode{
        boolean isWord;
        TrieNode[] children;

        TrieNode(){
            children = new TrieNode[26];
        }
    }

    private TrieNode constructTrie(String[] words){
        TrieNode root = new TrieNode();
        for (String w: words){
            TrieNode tn = root;
            for (int i = 0; i < w.length(); ++i){
                int off = w.charAt(i) - 'a';
                if (tn.children[off] == null)
                    tn.children[off] = new TrieNode();
                tn = tn.children[off];
            }
            tn.isWord = true;
        }
        return root;
    }
    private final int[][] toff = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    private void findHelper(char[][] board, TrieNode root, int i, int j, StringBuilder sb, List<String> res){
        if (root.isWord){
            res.add(sb.toString());
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || (board[i][j]&256) != 0 || root.children[board[i][j] - 'a'] == null ) //256 needs to be checked before last
            return;

        sb.append(board[i][j]);
        root = root.children[board[i][j] - 'a'];
        board[i][j] ^= 256; //this needs to happen after the first two steps
        for (int k = 0; k < toff.length; ++k){
            int x = i + toff[k][0], y = j + toff[k][1];
            findHelper(board, root, x, y, sb, res);
        }
        sb.deleteCharAt(sb.length() - 1);
        board[i][j] ^= 256;
    }

    //213
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0]; //this is a must otherwise the below len - 2, or start 1 will ArrayIndexOutofBounds
        return Math.max(robHelper(nums, 0, nums.length - 2), robHelper(nums, 1, nums.length - 1));
    }

    private int robHelper(int[] nums, int s, int e){
        int[] dp = new int[2];
        dp[1] = nums[s];
        int lsum = nums[s];
        for (int i = s + 1; i <= e; ++i){
            lsum = Math.max(dp[1], dp[0] + nums[i]);
            dp[0] = dp[1];
            dp[1] = lsum;
        }
        return lsum;
    }

    //218
    public List<int[]> getSkyline(int[][] buildings) {
        List<int[]> res = new ArrayList<>(); //2d coordinates uses List<int[]> is easier
        if (buildings == null || buildings.length == 0 ||buildings[0].length != 3)
            return res;
        //the thought is first translate m*3 array to (x,y) coordinates. because output we need that. and matters is left and right top cornors .
        //so we get all that and sort the coordinates by start. Then at every incoming points, we use a maxheap to store the height.
        //if we got new height on a new left, we output. we remove the height from the heap when on a right cornor, and if we meet a new low, we output.
        //so we need to check the peek of the heap and check with previous height and output a new res on any incoming coordinates.
        //to tell left or right. we use -height for a left and height for a right. PriorityQueue support remove(object) which is a linear time search!!!
        List<int[]> tmp = new ArrayList<>();
        for (int[] b : buildings){
            //left use -h, right use +h
            tmp.add(new int[]{b[0], -b[2]});
            tmp.add(new int[]{b[1], b[2]});
        }
        //sort by start. when start the same. meaning either two building start/end at same time, so lower out first(wont generate an output).
        Collections.sort(tmp, (i1, i2)->(i1[0] == i2[0]? i1[1] - i2[1]: i1[0] - i2[0]));
        //use maxheap to help give the highest building
        Queue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        pq.offer(0); //for last node
        int pre = 0;
        for (int[] t: tmp){
            if (t[1] < 0)
                pq.offer(-t[1]); //left node
            else
                pq.remove(t[1]); //right node linear time remove from pq
            int cur = pq.peek(); //current hightest
            if (cur != pre){
                res.add(new int[]{t[0], cur});
                pre = cur;
            }
        }
        return res;
    }

    //220
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (nums == null || nums.length == 0 || k <= 0 || t < 0)//k distinct index t - diff btw value
            return false;
        //keep a moving window of size k storing forward k neighbours. the reason is if on the right side, when we arrive there and look left, it will still be counted. so we only look left.
        //and this window should : given a x, find the nearest two numbers <= and >= x, and check them within the given t window --> we use TreeSet to so this
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i < nums.length; ++i){
            //only need to look into a window = k on the left
            Integer floor = ts.floor(nums[i]);
            Integer ceiling = ts.ceiling(nums[i]);
            if ((floor != null && floor + t >= nums[i]) || (ceiling != null && nums[i] + t >= ceiling))
                return true;
            ts.add(nums[i]);
            if (ts.size() > k)
                ts.remove(nums[i - k]);
        }
        return false;
    }

    //221
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int[][] dp = new int[matrix.length+1][matrix[0].length+1]; //enlarge by one to bypass the work of edges
        int res = 0;
        for (int i = 1; i < dp.length; ++i){
            for (int j = 1; j < dp[0].length; ++j){
                if (matrix[i-1][j-1] == '0')
                    dp[i][j] = 0;
                else
                    dp[i][j] = Math.min(dp[i-1][j], Math.min(dp[i][j-1], dp[i-1][j-1])) + 1;
                res = Math.max(res, dp[i][j]); //we want to catch max along the way
            }
        }
        return res * res; //we want the size. so square
    }

    //222
    public int countNodes(TreeNode root) { //Complexity is (logn) * (logn)
        if (root == null)
            return 0;
        int lh = 0, rh = 0;
        TreeNode t = root;
        while (t != null){
            ++lh;
            t = t.left;
        }
        t = root;
        while (t != null){
            ++rh;
            t = t.right;
        }
        if (lh == rh)
            return (1 << lh) - 1;
        return 1 + countNodes(root.left) + countNodes(root.right);

    }

    //224
    public int calculate(String s) {
        if (s == null || s.length() == 0) //Note this question restricts non-negative number -(-2+1)-3 not considered becuase -2 is neg not a minus
            return 0;
        Deque<Integer> st = new ArrayDeque<>();
        int res = 0;// ignore overflow
        //all op push a new x = +/-1 * peek(), all num pop one x. so need outside guard 1. and need a 1 to do the first num.
        //extra 1 or -1 in stack no problem because every number is calculated and done. nothing revisited.
        st.push(1);
        st.push(1);

        for (int i = 0; i < s.length(); ++i){ //use for loop all if cases need ++i
            if (Character.isSpaceChar(s.charAt(i)))
                continue;
            else if (Character.isDigit(s.charAt(i))) {
                int op = 0;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    op = op * 10 + s.charAt(i) - '0';
                    ++i;
                }
                res += st.pop() * op;
                --i;
            }
            else if (s.charAt(i) == ')')
                st.pop();
            else { //+-(
                st.push(st.peek() * (s.charAt(i) == '-'? -1: 1));
            }
        }
        return res;
    }

    //229
    public List<Integer> majorityElement(int[] nums) {
        //At most 2 elem can be > 1/3 first run voting algorithm then verify it they are
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        int m1 = -1, n1 = 0, m2 = -1, n2 = 0;
        for (int i : nums){
            if (i == m1)
                ++n1;
            else if (i == m2)
                ++n2;
            else if (n1 == 0){
                m1 = i;
                n1 = 1;
            }
            else if (n2 == 0){
                m2 = i;
                n2 = 1;
            }
            else {
                --n1;
                --n2;
            }
        }
        n1 = n2 = 0;
        for (int i : nums){
            if (i == m1) ++n1;
            else if (i == m2) ++n2;
        }
        if (n1 > nums.length / 3)
            res.add(m1);
        if (n2 > nums.length / 3)
            res.add(m2);
        return res;
    }

    //230
    private Integer kth;
    private int kthId;
    public int kthSmallest(TreeNode root, int k) {
        if (root == null || k < 1)
            return -1;
        kthSmallestHelper(root, k);
        return kth.intValue(); //Integer.intValue() returm primitive int
    }

    private void kthSmallestHelper(TreeNode root, int k){
        if (root == null)
            return;
        if (kth == null)
            kthSmallestHelper(root.left, k);
        if (++kthId == k)
            kth = root.val;
        if (kth == null)
            kthSmallestHelper(root.right, k);
    }

    //231
    public boolean isPowerOfTwo(int n) {
        return (n > 0) && (n & (n-1)) == 0;
    }

    //242
    public boolean isAnagram(String s, String t) {
        if (s == null)
            return t == null;
        if (t == null)
            return false;
        if (s.length() != t.length())
            return false;
        char[] sa = s.toCharArray();
        char[] ta = t.toCharArray();
        Arrays.sort(sa);
        Arrays.sort(ta);
        return Arrays.equals(sa, ta); //Note: Arrays.equals(Object[] a, Object[] b) will check a.equals(b) for single element!!
    }

    //246
    public boolean isStrobogrammatic(String num) {
        if (num == null || num.length() == 0)
            return false;
        int l = 0, r = num.length() - 1;
        Map<Character, Character> hm = new HashMap<>();
        hm.put('0', '0');
        hm.put('1', '1');
        hm.put('8', '8');
        hm.put('6', '9');
        hm.put('9', '6');
        while (l < r){
            if (!hm.containsKey(num.charAt(l)) || num.charAt(r) != hm.get(num.charAt(l)))
                return false;
            ++l;
            --r;
        }
        if (num.length() %2 == 1 && "018".indexOf(num.charAt(l)) < 0)
            return false;
        return true;
    }

    //247
    public List<String> findStrobogrammatic(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0)
            return res;
        findHelper(new char[n], 0, n - 1, res);
        return res;
    }
    private final char[][] pairs = {{'0', '0'}, {'1', '1'},{'8', '8'},{'6', '9'},{'9', '6'}};
    private final int SAME = 3;

    private void findHelper(char[] combi, int l, int r, List<String> res){
        if (l > r){ //output
            res.add(new String(combi));
            return;
        }
        if (l < r){ //loop through pairs
            for (char[] p : pairs){
                if (l == 0 && p[0] == '0')
                    continue;
                combi[l] = p[0];
                combi[r] = p[1];
                findHelper(combi, l + 1, r - 1, res);
            }
        }
        if (l == r){
            for (int i = 0; i < SAME; ++i){
                combi[l] = pairs[i][0];
                findHelper(combi, l + 1, r - 1, res);
            }
        }
    }

    //248
    public int strobogrammaticInRange(String low, String high) {
        //loop based on the low - high's size then generate number of the size, using char[]. when find a combi, check if valid
        if (low == null || low.length() == 0 || high == null || high.length() == 0)
            return 0;
        int res = 0;
        for (int s = low.length(); s <= high.length(); ++s)
            res+= findHelper(new char[s], 0, s - 1, low, high);
        return res;
    }
    private final char[][] pairs2 = {{'0','0'}, {'1', '1'}, {'8', '8'}, {'6', '9'}, {'9','6'}};
    private final int SAME2 = 3;
    private int findHelper(char[] combi, int l, int r, String low, String high){
        int res = 0;
        if (l > r){ //output
            String s = new String(combi);
            if ((s.length() == low.length() && s.compareTo(low) < 0) || (s.length() == high.length() && s.compareTo(high) > 0))
                return 0;
            return 1;
        }
        if (l < r){
            for (char[] p : pairs2){
                if (l == 0 && p[0] == '0')
                    continue;
                combi[l] = p[0];
                combi[r] = p[1];
                res += findHelper(combi, l + 1, r - 1, low, high);
            }
        }
        else { //l == r
            for (int i = 0; i < SAME; ++i){
                combi[l] = pairs[i][0];
                res += findHelper(combi, l + 1, r - 1, low, high);
            }
        }
        return res;
    }

    //251
    public class Vector2D implements Iterator<Integer> {
        List<Iterator<Integer>> iters;
        int idx;
        public Vector2D(List<List<Integer>> vec2d) {
            iters = new ArrayList<>();
            for (List<Integer> l : vec2d){
                if (!l.isEmpty())
                    iters.add(l.iterator());
            }
        }

        @Override
        public Integer next() {
            return iters.get(idx).next();
        }

        @Override
        public boolean hasNext() {
            if (idx == iters.size())
                return false;
            if (iters.get(idx).hasNext())
                return true;
            if (++idx == iters.size())
                return false;
            return true;
        }
    }

    //252
    public boolean canAttendMeetings(Interval[] intervals) {
        if (intervals == null || intervals.length <= 1)
            return true;
        Arrays.sort(intervals, (i1, i2)->i1.start - i2.start);
        for (int i = 1; i < intervals.length; ++i){
            if (intervals[i].start < intervals[i-1].end)
                return false;
        }
        return true;
    }

    //253
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;
        Arrays.sort(intervals, (i1, i2)->i1.start == i2.start? i1.end - i2.end: i1.start - i2.start);
        //Queue<Interval> pq = new PriorityQueue<>((i1, i2)-> i1.end - i2.end);
        Queue<Integer> pq = new PriorityQueue<>(); //NOTE: here we can safely just use the end time as PQ. we know start already sorted!
        pq.offer(intervals[0].end);
        for (int i = 1; i < intervals.length; ++i){
            if (intervals[i].start >= pq.peek())
                pq.poll();
            pq.offer(intervals[i].end);
        }
        return pq.size();
    }

    //254
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < 4)
            return res;
        getFactorsHelper(n, 2, new ArrayList<Integer>(), res);
        return res;
    }

    private void getFactorsHelper(int n, int start, List<Integer> combi, List<List<Integer>> res){
        if (n == 1){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int i = start; i * i <= n; ++i){ //HERE MUST PASS START AND START AT START, WE NEED TO MAKE SURE ONLY MULTIPLY A LARGER NUMBER!!!
            if (n % i == 0){
                combi.add(i);
                getFactorsHelper(n / i, i, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
        if (combi.size() > 0){
            combi.add(n);
            getFactorsHelper(1, n, combi, res);
            combi.remove(combi.size() - 1);
        }
    }

    //261
    public boolean validTree(int n, int[][] edges) {
        //graph -> valid tree: acyclic connected. connected: n nodes should have n-1 edges. However if there is cycle. dup edges may fake the # of edges. but then topo sort count will !=n
        //so we just check edges count = n -1 in the beginning, then bfs topo sort and at the end check if cnt == n. this is enough.
        if (n <= 0 || edges == null || edges.length != n-1) //for no cycle case, this filter out non-connected case
            return false;
        if (n == 1 && edges.length == 0)
            return true; //note we start from indegree = 1 nodes. so single root node case should be seperately checked here
        //construct graph
        int[] indegree = new int[n];
        List<Integer>[] children = new List[n];
        for (int i = 0; i < children.length; ++i)
            children[i] = new ArrayList<>();
        for (int[] e : edges){
            ++indegree[e[0]];
            ++indegree[e[1]];
            children[e[0]].add(e[1]);
            children[e[1]].add(e[0]);
        }
        //BFS
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegree.length; ++i){
            if (indegree[i] == 1) {
                queue.offer(i);
            }
        }
        int cnt = 0;
        while (!queue.isEmpty()){
            int x = queue.poll();
            ++cnt;
            for (int c : children[x]){
                if (--indegree[c] == 1){
                    queue.offer(c);
                }
            }
        }
        return cnt == n;
    }

    //271
    public class Codec { //use len#str to encode --> no special

        // Encodes a list of strings to a single string.
        public String encode(List<String> strs) {   //suppose original string have a string "3#" , it will still be encoded. as "2#3#"
            if (strs == null || strs.size() == 0)
                return "";
            StringBuilder sb = new StringBuilder();
            for (String s : strs){
                sb.append(s.length());
                sb.append("#");
                sb.append(s);
            }
            return sb.toString();
        }

        // Decodes a single string to a list of strings.
        public List<String> decode(String s) { // when decode, we will always meet the first 2# and decode 3# as a normal string then bypass it. so there is no problem.
            List<String> res = new ArrayList<>();
            if (s == null || s.length() == 0)
                return res;
            int start = 0, pound = 0;
            while (start < s.length()){
                pound = s.indexOf('#', start);
                int len = Integer.parseInt(s.substring(start, pound));
                start = pound + len + 1; //new start
                res.add(s.substring(pound + 1, start));
            }
            return res;
        }
    }

    //279
    public int numSquares(int n) {
        if (n < 1)
            return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, n);
        dp[0] = 0;
        for (int i = 0; i < dp.length - 1; ++i){
            for (int j = 1; i + j * j < dp.length; ++j){
                dp[i + j * j] = Math.min(dp[i + j * j], dp[i] + 1);
            }
        }
        return dp[dp.length - 1];
    }

    //281
    public class ZigzagIterator {
        List<Iterator<Integer>> iters;
        int idx;
        public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
            iters = new ArrayList<>();
            if (v1.size() != 0)
                iters.add(v1.iterator()); //first we only add non-empty ones
            if (v2.size() != 0)
                iters.add(v2.iterator());
        }

        public int next() {
            int x = iters.get(idx).next();
            idx = (idx+1) % iters.size();
            return x;
        }

        public boolean hasNext() {
            while (iters.size() > 0 && !iters.get(idx).hasNext()){ //multiples can already run out so a while loop
                iters.remove(idx);
                idx = (idx == iters.size()? 0: idx); //if we are in a middle, we stay. otherwise if we == size. we go to 0
            }
            if (iters.size() == 0)
                return false;
            return true;
        }
    }

    //288
    public class ValidWordAbbr {
        Map<String, Integer> hmWord;
        Map<String, Integer> hmAbbr;

        public ValidWordAbbr(String[] dictionary) {
            hmWord = new HashMap<>();
            hmAbbr = new HashMap<>();
            if (dictionary == null || dictionary.length == 0)
                return;
            for (String s : dictionary){
                hmWord.put(s, hmWord.getOrDefault(s, 0) + 1);
                String a = getAbbr(s);
                hmAbbr.put(a, hmAbbr.getOrDefault(a, 0) + 1);
            }
        }

        private String getAbbr(String s){
            if (s == null || s.length() <= 2) //Note java Set interface allow null as key
                return s;
            StringBuilder sb = new StringBuilder();
            sb.append(s.charAt(0)); //char + int + char is a int. must have at least one string.
            sb.append(s.length() -2);
            sb.append(s.charAt(s.length() - 1));
            return sb.toString();
        }

        public boolean isUnique(String word) {
            String a = getAbbr(word);
            return !hmAbbr.containsKey(a) || (hmWord.containsKey(word) && hmWord.get(word) == hmAbbr.get(a));
        }
    }

    //295
    public class MedianFinder {
        private Queue<Integer> minq = new PriorityQueue<>();
        private Queue<Integer> maxq = new PriorityQueue<>(Collections.reverseOrder());

        // Adds a number into the data structure.
        public void addNum(int num) {
            if (maxq.isEmpty() || num < maxq.peek())
                maxq.offer(num);
            else
                minq.offer(num);
            if (minq.size() > maxq.size()) //balance when inserting
                maxq.offer(minq.poll());
            else if (maxq.size() > minq.size() + 1)
                minq.offer(maxq.poll());
        }

        // Returns the median of current data stream
        public double findMedian() {
            if (maxq.isEmpty())
                return -1;
            if (maxq.size() == minq.size() + 1)
                return maxq.peek();
            return (minq.peek() + maxq.peek())/2.0;
        }
    };

    //296
    public int minTotalDistance(int[][] grid) {
        //Find the median spot in both x and y. then loop and count the dist
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        List<Integer> listx = new ArrayList<>();
        List<Integer> listy = new ArrayList<>();
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 1){
                    listx.add(i);
                    listy.add(j);
                }
            }
        }
        Collections.sort(listy);
        int midx = listx.get(listx.size()/2);
        int midy = listy.get(listy.size()/2);
        int res = 0;
        for (int i = 0; i < listx.size(); ++i){
            res += Math.abs(listx.get(i) - midx);
            res += Math.abs(listy.get(i) - midy);
        }
        return res;
    }

    //297
    public class Codec2 {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null)
                return "#";
            StringBuilder sb = new StringBuilder();
            serialHelper(root, sb);
            return sb.toString();
        }

        private void serialHelper(TreeNode root, StringBuilder sb){
            if (root == null){
                sb.append("#");
                sb.append(",");
                return;
            }
            sb.append(root.val);
            sb.append(",");
            serialHelper(root.left, sb);
            serialHelper(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.length() == 0)
                return null;
            String[] tokens = data.split(",");
            Queue<String> queue = new LinkedList<>();
            for (String t : tokens) //Here LinkedList(Collections c)  Does not have ctor accepting (T [] array) only collecionts!!
                queue.offer(t);
            return deserializeHelper(queue);
        }

        private TreeNode deserializeHelper(Queue<String> queue){
            if (queue.isEmpty())
                return null;
            String s = queue.poll();
            if (s.equals("#"))
                return null;
            int x = Integer.parseInt(s);
            TreeNode root = new TreeNode(x);
            root.left = deserializeHelper(queue);
            root.right = deserializeHelper(queue);
            return root;
        }
    }

    //298
    private int longestconsec = 1;
    public int longestConsecutive(TreeNode root) {
        if (root == null)
            return 0;
        longestConsecHelper(root.left, root.val, 1);
        longestConsecHelper(root.right, root.val, 1);
        return this.longestconsec;
    }

    private void longestConsecHelper(TreeNode root, int pre, int len){
        if (root == null)
            return;
        if (root.val == pre + 1){
            len += 1;
            this.longestconsec = Math.max(longestconsec, len);
        }
        else
            len = 1;
        longestConsecHelper(root.left, root.val, len);
        longestConsecHelper(root.right, root.val, len);
    }
}
