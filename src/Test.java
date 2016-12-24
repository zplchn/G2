import java.util.*;

/**
 * Created by zplchn on 12/10/16.
 */
public class Test {

    public static void main(String[] args){
        Solution s = new Solution();
//        System.out.println("aaa" + 'b');
//        System.out.println("aaa" + 28);
//        System.out.println("aaa" + 28 + 2); //aaa282
//        System.out.println(Character.getNumericValue('A'));

//        int x = -2147483648, y = 1;
//        System.out.println((x ^ y) >>> 31);
//        boolean isNeg = ((x ^ y) >>> 31) == 1;
//        System.out.println(isNeg);

        //String kk = s.fractionToDecimal(-2147483648, 1);
        //System.out.println(kk);
//        Queue<Integer> queue = new LinkedList<>();
//        queue.offer(1);
//        queue.offer(2);
//        queue.remove(1);
//        System.out.println(queue.poll());
//        String str = "/h/r/";
//        for (String k : str.split("/"))
//            System.out.println("[" + k + "]");
//        String sss = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext";
//        s.lengthLongestPath(sss);

//        int [] nums = {3,4,6,5};
//        for (int i = 1; i <= 4; ++i)
//            System.out.println(Arrays.toString(maxNum(nums, i)));
        System.out.println(getLength(2,5));
        String str = "aaa";
        String str5 = "aaaaa";
        String str4 = "abcdabcd";
        System.out.println(encode(str));
        System.out.println(encode(str5));
        System.out.println(encode(str4));



    }

    // find the max number that can be formed in the nums, with i digits.
    // use greedy的方法去找这个数。
    public static int[] maxNum(int[] nums, int k) {
        int[] res = new int[k];
        int j = 0, n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            // 因为要保证顺序，所以n - i > k - j, 同时，如果发现之前那一位比现在这一位要小的话，就replace成new value
            while (n - i + j > k && j > 0 && res[j - 1] < nums[i]) j--;
            if (j < k) res[j++] = nums[i];
        }
        return res;
    }

//    public static char compress(String str){
//        char[] ca = str.toCharArray();
//
//    }
public static boolean checkRepeating(String s, int l, int r, int start, int end){
    if( (end-start) % (r-l) != 0 ){
        return false;
    }
    int len = r-l;
    String pattern = s.substring(l, r);
    for(int i = start; i +len <= end; i+=len){
        if(!pattern.equals(s.substring(i, i+len))){
            return false;
        }
    }
    return true;
}

    public static int getLength(int plen, int slen){
        return (int)(Math.log10(slen/plen)+1);
    }

    public static String encode(String s){
        int len = s.length();
        int[][] dp = new int[len+1][len+1];

        for(int i = dp.length - 1; i >= 0; --i){
            for(int j = i; j < dp[0].length; ++j){
                dp[i][j] = j - i;
            }
        }

        Map<String, String> dpMap = new HashMap<>();

        for(int i = dp.length - 1; i >= 0; --i){
            for(int j = i+1; j < dp[0].length; ++j){

                String temp = s.substring(i, j);
                if(dpMap.containsKey(temp)){
                    dp[i][j] = dpMap.get(temp).length();
                    continue;
                }
                String ans = temp;
                for(int k = i+1; k < j; ++k){
                    String leftStr = s.substring(i, k);
                    String rightStr = s.substring(k, j);
                    if(dp[i][j] > dp[i][k] + dp[k][j]){
                        dp[i][j] = dp[i][k] + dp[k][j];
                        ans = dpMap.get(leftStr) + dpMap.get(rightStr);
                    }
                    if( checkRepeating(s, i, k, i, j)) {
                        int repeat = (j - i) / (k - i);
                        int newSize = Integer.toString(repeat).length() + 2 + dp[i][k];
                        if (newSize < dp[i][j]) {
                            dp[i][j] = newSize;
                            ans = repeat + "[" + dpMap.get(leftStr) + "]";
                        }
                    }
                }
                dpMap.put(temp,ans);
            }
        }
        return dpMap.get(s);
    }




}
