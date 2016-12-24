import java.util.*;

/**
 * Created by zplchn on 12/11/16.
 */
public class Solution400 {

    //400
    public int findNthDigit(int n) {
        if (n < 1)
            return -1;
        //1. first figure out n falls into which interval of total length of digits 1-9 -> 9; 10-99->90; 100-999->900
        //2. then get the number on the index of the n in the certain interval n / len of digits of the interval
        //3. then convert the number to string and get the char on the bit of n % len

        int len = 1, start = 1;//every iteration need to adjust to 1-based
        long cnt = 9; //cnt could overflow

        while (n > len * cnt){
            n -= len * cnt;
            ++len;
            cnt *= 10; //cnt is #of num in the interval 9 , 90, 900, and need to multiply total digits len 1,2,3
            start *= 10;
        }
        //now n is the 1-based offset
        int num = start + (n - 1) / len;
        return String.valueOf(num).charAt((n - 1) % len) - '0';
    }

    //401
    public List<String> readBinaryWatch(int num) {
        List<String> res = new ArrayList<>();
        if (num < 0)
            return res;
        //Since combination of hh:mm is finite, so instead of doing combination from num, loop through just all hh:mm and get count of bit
        for (int h = 0; h < 12; ++h){
            for (int m = 0; m < 60; ++m){
                if (Integer.bitCount((h << 6) | m) == num) //shift h makes it a whole number with two parts no-overlap no extra space between
                    res.add(String.format("%d:%02d", h, m));
            }
        }
        return res;
    }

    //402
    public String removeKdigits(String num, int k) {
        if (num == null || k <= 0 || num.length() == 0)
            return num;
        if (k >= num.length())
            return "0";
        Deque<Character> dq = new ArrayDeque<>();
        int tempK = k; //need to save k because k will decrease but we need cut at the end to n - k
        for (int i = 0; i < num.length(); ++i){
            char c = num.charAt(i);
            while (k > 0 && !dq.isEmpty() && c < dq.peek()){
                dq.pop();
                --k;
            }
            dq.push(c); //if more ascending, the stack will just have all of them
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < num.length() - tempK; ++i) //need cut off at n-k
            sb.append(dq.pollLast());
        while (sb.length() > 1 && sb.charAt(0) == '0') //note must check length along with leading 0. when there is only a 0
            sb.deleteCharAt(0); // need to handle 10200 case
        return sb.toString();
    }

    //404
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null)
            return 0;
        leftHelper(root, false); //Note: here single node is a leaf. But it cannot say it's a Left leaf so is excluded from result!!!
        return this.lsum;
    }
    private int lsum;

    private void leftHelper(TreeNode root, boolean isLeft){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            if (isLeft)
                this.lsum += root.val;
            return;
        }
        leftHelper(root.left, true);
        leftHelper(root.right, false);
    }

    //406
    public int[][] reconstructQueue(int[][] people) {
        //so we first sort the array descending based on h, tall first, short after, then we insert based on the k, k should be index in new list
        if (people == null || people.length == 0 || people[0].length == 0)
            return people;
        Arrays.sort(people, (a1, a2)->(a1[0] == a2[0]? a1[1] - a2[1]: a2[0] - a1[0])); //h descending, k asending
        List<int[]> tmp = new LinkedList<>();
        for (int[] i : people)
            tmp.add(i[1], i);

        return tmp.toArray(new int[tmp.size()][]);
    }

    //407
    public int trapRainWater(int[][] heightMap) {
        if (heightMap == null || heightMap.length == 0 || heightMap[0].length == 0)
            return 0;
        //find outside bank like flood fill. always find the minimum height being the bank. if neighbours lower than it, can store water. because all the rest are higher banks.
        //moved into q with the height of the higher of water level or the bar itself. BFS requires visited map.
        int res = 0;
        //bfs from the outer 4 edges. but everytime we 'd like to find the lowest bar. use pq. int[i, j, z]
        Queue<int[]> pq = new PriorityQueue<>((i1, i2)-> i1[2] - i2[2]);
        boolean[][] visited = new boolean[heightMap.length][heightMap[0].length];
        //first put all 4 edges into queue
        for (int j = 0; j < heightMap[0].length; ++j){
            pq.offer(new int[]{0, j, heightMap[0][j]});
            pq.offer(new int[]{heightMap.length - 1, j, heightMap[heightMap.length - 1][j]});
            visited[0][j] = true;
            visited[heightMap.length - 1][j] = true;
        }
        for (int i = 1; i < heightMap.length - 1; ++i){
            pq.offer(new int[]{i, 0, heightMap[i][0]});
            pq.offer(new int[]{i, heightMap[0].length -1, heightMap[i][heightMap[0].length -1]});
            visited[i][0] = true;
            visited[i][heightMap[0].length -1] = true;
        }
        int[][] off = {{-1, 0}, {1, 0}, {0, -1}, {0,1}};
        while (!pq.isEmpty()){
            int[] t = pq.poll();
            for (int[] o : off){
                int x = t[0] + o[0], y = t[1] + o[1];
                if (x >= 0 && x < heightMap.length && y >= 0 && y < heightMap[0].length && !visited[x][y]){
                    int diff = t[2] - heightMap[x][y];
                    res += diff > 0? diff: 0;
                    visited[x][y] = true;
                    pq.offer(new int[]{x, y, diff >= 0? t[2]: heightMap[x][y]});
                }
            }
        }
        return res;
    }

    //408
    public boolean validWordAbbreviation(String word, String abbr) {
        if (word == null || abbr == null)
            return false;
        int i = 0, j = 0, jump = 0;
        while (i < word.length() && j < abbr.length()) {
            if (Character.isDigit(abbr.charAt(j))) {
                if (jump == 0 && abbr.charAt(j) == '0') //Note: when multi-char string number, always check leading zero is validity
                    return false;
                jump = jump * 10 + abbr.charAt(j) - '0';
            }
            else {
                i += jump;
                jump = 0;
                if (i >= word.length() || word.charAt(i) != abbr.charAt(j))
                    return false;
                else
                    ++i;
            }
            ++j;
        }
        i += jump; // "OK -> "o1"
        return i == word.length() && j == abbr.length();
    }

    //409
    public int longestPalindrome(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int[] cnt = new int[128];
        for (int i = 0; i < s.length(); ++i){
            ++cnt[s.charAt(i)];
        }
        int res = 0, odd = 0;
        for (int v : cnt){
            if (v % 2 == 0)
                res += v;
            else {
                odd = 1;
                res += v - 1; //at least 3 -> 2, 5->4 can contribute partial of that
            }
        }
        return res + odd;
    }

    //414
    public int thirdMax(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : nums){
            ts.add(i);
            if (ts.size() > 3)
                ts.pollFirst();
        }
        return ts.size() == 3? ts.pollFirst(): ts.pollLast(); //Noe the pollFirst(), pollLast()
    }

    //415
    public String addStrings(String num1, String num2) {
        if (num1 == null || num1.length() == 0)
            return num2;
        if (num2 == null || num2.length() == 0)
            return num1;
        int i = num1.length() - 1, j = num2.length() - 1, carry = 0;
        StringBuilder sb = new StringBuilder();
        while (i >= 0 || j >= 0 || carry != 0){
            int sum = (i >= 0? num1.charAt(i--) - '0': 0) + (j >= 0? num2.charAt(j--) - '0': 0) + carry;
            carry = sum / 10;
            sb.append(sum % 10);
        }
        return sb.reverse().toString();
    }

    //417
    public List<int[]> pacificAtlantic(int[][] matrix) {
        //Floodfill from the pacific edges and check a dfs path exists to the other side
        List<int[]> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return res;
        //start from left and top dfs mark visited
        boolean [][] toPacific = new boolean[matrix.length][matrix[0].length];
        boolean [][] toAtlantic = new boolean[matrix.length][matrix[0].length];
        for (int j = 0; j < matrix[0].length; ++j) {
            toPacific[0][j] = true;
            pathHelper(matrix, 0, j, 0, toPacific);
            toAtlantic[matrix.length - 1][j] = true;
            pathHelper(matrix, matrix.length - 1, j, 0, toAtlantic);
        }
        for (int i = 0; i < matrix.length; ++i) {
            toPacific[i][0] = true;
            pathHelper(matrix, i, 0, 0, toPacific);
            toAtlantic[i][matrix[0].length - 1] = true;
            pathHelper(matrix, i, matrix[0].length - 1, 0, toAtlantic);
        }

        for (int i = 0; i < matrix.length; ++i){
            for (int j = 0; j < matrix[0].length; ++j){
                if (toPacific[i][j] && toAtlantic[i][j])
                    res.add(new int[]{i, j});
            }
        }
        return res;
    }
    private final int[][] poff = {{-1, 0},{1, 0},{0, -1},{0, 1}};
    private void pathHelper(int[][] matrix, int i, int j, int pre, boolean[][] toOcean){
        for (int k = 0; k < poff.length; ++k){
            int x = i + poff[k][0], y = j + poff[k][1];
            if (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length && !toOcean[x][y] && matrix[x][y] >= matrix[i][j]) {
                toOcean[x][y] = true;
                pathHelper(matrix, x, y, matrix[i][j], toOcean);
            }
        }
    }

    //418
    public int wordsTyping(String[] sentence, int rows, int cols) {
        if (sentence == null || sentence.length == 0 || rows < 1 || cols < 1)
            return 0;
        //concat all strings with space in between
        String str = String.join(" ", sentence) + " "; //String.join(delimitnator, Iterable<CharSequence>) !!
        int s = 0; // in the end. we count how many times s / len taht will be the repeat factor
        for (int i = 0; i < rows; ++i){
            //as long as we r not at edge. we just repeat. so we add the col. no prolbem
            s += cols;
            //deal with edge cases
            if (str.charAt(s % str.length()) == ' ')
                ++s;
            else {
                while (s > 0 && str.charAt((s - 1) % str.length()) != ' ') // note here if a word is widder than a given col. it will backoff till to 0. and will repeat not filling but row goes to max numberm
                    --s;
            }
        }
        return s/str.length();
    }

    //421
    public int findMaximumXOR(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = 0, leadingMask = 0;
        //since we need to know the mask, we loop by bit from 31 to 0. at each step, we extract the prefix by AND a all 1 mask till the current bit
        // now we use a temp = res | 1 at current bit. we use this to xor everyone in the prefix set.
        // if a ^ temp = b, then temp = a ^ b. so if any a from the set xor the temp and result b exist in the set. that means, this
        //a ^ b = temp and temp is always the max achieveable result we need to find.
        for (int i = 31; i >= 0; ++i){
            leadingMask |= (1 << i);
            //since we need to check given a A, if a b exists in the same set. we use a hashset
            Set<Integer> hs = new HashSet<>();
            for (int x : nums)
                hs.add(x & leadingMask); //note here is a AND, extract the prefix
            int t = res | (1 << i); //t is the max achievable result and we need to test if it's contructable by two number in the set
            for (int x : hs){
                if (hs.contains(x ^ t)) { //if x ^ t = y then x ^ y = t; and meaning t is achievable
                    res = t;
                    break;
                }
            }
        }
        return res;
    }

    //422
    public boolean validWordSquare(List<String> words) {
        //word[i][j] check == word[j][i] so need to check IndexOutOfBounds exception. cannot just check half diagnal because some row can go very long > words.size()  [aa, aaxxxxxx].
        if (words == null)
            return false;
        for (int i = 0; i < words.size(); ++i){
            for (int j = 0; j < words.get(i).length(); ++j){
                if (j >= words.size() || i >= words.get(j).length() || words.get(i).charAt(j) != words.get(j).charAt(i))
                    return false;
            }
        }
        return true;
    }


    //425
    class TrieNode { //For prefix tree that frequently given a prefix, return all possible descendants
                     //store an extra List<String> startsWith and immediately return
        List<String> startsWith; //if we want to check startsWith. we use boolean isWord. but here need return a lit
        TrieNode[] children;

        TrieNode(){
            startsWith = new ArrayList<>();
            children = new TrieNode[26];
        }
    }

    class Trie {
        TrieNode root;

        Trie(){}
        Trie(String[] words){
            root = new TrieNode();

            for (String s : words){
                TrieNode tr = root;
                tr.startsWith.add(s);
                for (int j = 0; j < s.length(); ++j){
                    int off = s.charAt(j) - 'a';
                    if (tr.children[off] == null)
                        tr.children[off] = new TrieNode();
                    tr = tr.children[off];
                    tr.startsWith.add(s);
                }
            }
        }

        List<String> findByPrefix(String prefix){
            List<String> res = new ArrayList<>();
            TrieNode tr = root;
            for (int i = 0; i < prefix.length(); ++i){
                int off = prefix.charAt(i) - 'a';
                if (tr.children[off] == null)
                    return res;
                tr = tr.children[off];
            }
            res.addAll(tr.startsWith);
            return res;
        }
    }

    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> res = new ArrayList<>();
        if (words == null || words.length == 0)
            return res;
        Trie trie = new Trie(words);
        int n = words[0].length();

        wordHelper(n, trie, new ArrayList<String>(), res);
        return res;
    }

    private void wordHelper(int n, Trie trie, List<String> combi, List<List<String>> res){
        if (combi.size() == n){
            res.add(new ArrayList<>(combi));
            return;
        }
        StringBuilder sb = new StringBuilder();
        int idx = combi.size();
        for (String s : combi){
            sb.append(s.charAt(idx));
        }
        List<String> startsWith = trie.findByPrefix(sb.toString());
        for (String s : startsWith){
            combi.add(s);
            wordHelper(n, trie, combi, res);
            combi.remove(combi.size() - 1);
        }
    }


    //435
    public int eraseOverlapIntervals(Interval[] intervals) {
        if (intervals == null || intervals.length <= 1)
            return 0;
        //this problem is same as scheduling intervals problem. find the max non-overlapping intervals. using greedy, get the first ending intervals's end, if overlap, pass, otherwise, set the new end.
        Arrays.sort(intervals, (i1, i2) -> (i1.end - i2.end)); //only need to sort end
        int earliestEnd = intervals[0].end, res = 0;
        for (int i = 1; i < intervals.length; ++i){
            if (intervals[i].start < earliestEnd)
                ++res; // an overlap
            else
                earliestEnd = intervals[i].end;
        }
        return res;
    }

    //436
    public int[] findRightInterval(Interval[] intervals) {
        //find bare minimum right entry >= self. it hints at a ceiling. so use TreeMap
        //put the interval's (start, index) to a tm. and for each one, find ceiling key's index to output array
        if (intervals == null)
            return null;
        int[] res = new int[intervals.length];
        TreeMap<Integer, Integer> tm = new TreeMap<>(); //not every map support initial cap. TreeMap NOT support!!!
        Map.Entry<Integer, Integer> entry;
        for (int i = 0; i < intervals.length; ++i)
            tm.put(intervals[i].start, i);
        for (int i = 0; i < intervals.length; ++i){
            entry = tm.ceilingEntry(intervals[i].end);
            res[i] = entry == null? -1: entry.getValue();
        }
        return res;
    }

    //447
    public int numberOfBoomerangs(int[][] points) {
        //the point is, given one point, there are k points the same distance to it, so they are on a circle with diameter=k. and the permutation numbers is A-K-2, K*(K-1)
        //so we start from every node and treat it as circle, and use a hm to store the diameters. then add up all the possible permutations

        if (points == null || points.length == 0 || points[0].length != 2)
            return 0;
        int res = 0;
        for (int[] p: points){
            //square root is identical of square here as negative is not possible for a distance
            Map<Integer, Integer> hm = new HashMap<>();
            for (int[] q: points){
                if (p[0] == q[0] && p[1] == q[1])
                    continue;
                int x = p[0] - q[0], y = p[1] - q[1];
                int dist = x * x + y * y;
                hm.put(dist, hm.getOrDefault(dist, 0) + 1);
            }
            //do permutation
            for (int v: hm.values())
                res += v * (v - 1); //A- v 2
        }
        return res;
    }

    //459
    public boolean repeatedSubstringPattern(String str) {
        if (str == null || str.length() <= 1)
            return false;
        //1. repeat part must start at 0. so we start at i and find substring till half n;
        //2. this substring's len should n % sub == 0, then we repeat n / sub and compare
        int n = str.length();
        for (int i = 1; i <= str.length()/2; ++i){
            if (n % i == 0){
                String sub = str.substring(0, i);
                int j = 0;
                while (j < str.length() && str.startsWith(sub, j))
                    j += sub.length();
                if (j == str.length())
                    return true;
            }
        }
        return false;
    }

    //463
    public int islandPerimeter(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int res = 0;
        //according to if a given node can be of the 4 edges to process
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 1){
                    if (j == 0 || grid[i][j-1] == 0) ++res; //left
                    if (j == grid[0].length - 1 || grid[i][j+1] == 0) ++res; //right
                    if (i == 0 || grid[i-1][j] == 0) ++res; //top
                    if (i == grid.length - 1 || grid[i+1][j] == 0) ++res; //btm
                }
            }
        }
        return res;
    }
}
