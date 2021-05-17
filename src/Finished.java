import java.util.*;

/*  //                        406. 根据身高重建队列
//    先排序，再插入，先把身高从大到小进行排序，然后再把k按照从小到大进行排序
//因为，k的意思是，前面身高大于等于我的个数，那么我先排序身高高的，按照k从小到大排序，那么再排序身高低的时候，也从小到大排序
//这个时候，只用考虑目前需要放的位置是否对就行了(也就是直接在list里面找第k位置就行了)，因为不会再影响之前已经放好的元素的k了
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people,new Comparator<int[]>(){
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[0]>o2[0]){
                    return -1;
                }
                else if(o1[0]<o2[0]){
                    return 1;
                }
                else {
                    return o1[1]-o2[1];
                }
            }
        });
        List<int[]> res=new ArrayList<>();
        for (int[] person : people) {
            res.add(person[1],person);
        }
        int[][] ans=new int[people.length][];
        for(int i=0;i<people.length;i++){
            ans[i]=res.get(i);
        }

        return ans;
    }*/
/**
 * 399. 除法求值
 * 经典的带权并查集 ，并查集的find的时候，parent更改了，所以，更改权重给的时候，要记得先保存之前parent是谁，再更新权重
 * 在union的时候，不需要储存谁高谁低，因为find的时候，就会直接变成两层，就一个进行union了，其实也是记忆化搜索，只要find一次，
 * 之后就是O（1）的复杂度了
 *
 */

    /*public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String,Integer> map = new HashMap<>();
        UnionFind unionFind = new UnionFind(equations.size() * 2);
        int id=0;
        for(int i=0;i<equations.size();i++){
            String x = equations.get(i).get(0);
            if(!map.containsKey(x)){
                map.put(x, id++);
            }
            String y=equations.get(i).get(1);
            if(!map.containsKey(y)){
                map.put(y, id++);
            }
            unionFind.union(map.get(x),map.get(y),values[i]);
        }

        double res[] =new double[queries.size()];
        for(int i=0;i<queries.size();i++){
            String x = queries.get(i).get(0);
            String y=queries.get(i).get(1);
            if(!map.containsKey(x)||!map.containsKey(y)){
                res[i]=-1.0;
            }
            else {
                res[i]=unionFind.calculate(map.get(x), map.get(y));
            }
        }

        return res;
    }
    //带权重的并查集
    private class UnionFind{
        int[] parent;
        double []weight;
        public UnionFind(int n){
            this.parent=new int[n];
            this.weight=new double[n];
            for(int i=0;i<n;i++){
                parent[i]=i;
                weight[i]=1.0;
            }
        }
        int find(int x){
            if(parent[x]==x){
                return x;
            }
            else {
                //parent路径压缩以后，之前的parent已经找不到了所以得先储存一下
                int origin=parent[x];
                parent[x]=find(parent[x]);
                weight[x]=weight[origin]*weight[x];
                return parent[x];
            }
        }
        void union(int i,int j,double value){
            int parenti = find(i);
            int parentj=find(j);
            if(parenti==parentj){
                return ;
            }
            else {
//                让i的父亲，指向j的父亲
                parent[parenti]=parentj;
                weight[parenti]=value*weight[j]/weight[i];
            }
        }
        double calculate(int x,int y){
            int px = find(x);
            int py = find(y);
            if(py!=px){
                return -1.0;
            }
            else {
//                查询的时候已经将weight设定为指向根节点的weight了
                return weight[x]/weight[y];
            }
        }
    }*/
/*   394. 字符串解码
    public String decodeString(String s) {
        StringBuffer ans=new StringBuffer();
        Deque<Integer> stackIntegers = new LinkedList<>();
        Deque<Character> stackChar=new LinkedList<>();
        Deque<StringBuffer> stackString=new LinkedList<>();
        stackString.push(ans);
        char[] array = s.toCharArray();
        for(int j=0;j<array.length;j++){
            Character c=array[j];
            if(Character.isDigit(c)){
                StringBuffer sb = new StringBuffer();
                while (Character.isDigit(array[j])){
                    sb.append(array[j]);
                    j++;
                }
                j--;
                stackIntegers.push(new Integer(sb.toString()));
            }
            else if(c.equals('[')){
                stackChar.push('[');
                stackString.push(new StringBuffer());
            }
            else if(c.equals(']')){
                Integer poll = stackIntegers.poll();
                stackChar.poll();
                StringBuffer temp = stackString.poll();
                StringBuffer peek = stackString.peek();
                for(int i=0;i<poll;i++){
                    peek.append(temp);
                }
            }
            else {
                StringBuffer sb=new StringBuffer();
                while (j<array.length&&array[j]!='['&&array[j]!=']'&&!Character.isDigit(array[j])){
                    sb.append(array[j]);
                    j++;
                }

                StringBuffer peek = stackString.peek();
                peek.append(sb);

                if(j==array.length) break;
                else {
                    j--;
                }

            }
        }
        return stackString.peek().toString();
    }*/
/*//   347. 前 K 个高频元素 固定大小的小顶堆，可以用来找最大值，如果要插入的值比当前的顶大，弹出顶，插入值，否则，就是，这个数，比堆中所有的数都小，那么舍弃他
    public int[] topKFrequent(int[] nums, int k) {
        int res[]=new int[k];
        Map<Integer,Integer> map=new HashMap<>();
        for (int num : nums) {
            if(map.containsKey(num)){
                map.put(num, map.get(num)+1);
            }
            else {
                map.put(num, 1);
            }
        }
        PriorityQueue<int []> priorityQueue=new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1]-o2[1];
            }
        });

        for (Map.Entry<Integer, Integer> integerIntegerEntry : map.entrySet()) {
            if(priorityQueue.size()<k){
                priorityQueue.add(new int[]{integerIntegerEntry.getKey(),integerIntegerEntry.getValue()});
            }
            else {
                if(integerIntegerEntry.getValue()>priorityQueue.peek()[1]){
                    priorityQueue.poll();
                    priorityQueue.add(new int[]{integerIntegerEntry.getKey(),integerIntegerEntry.getValue()});
                }
            }
        }
        for(int i=0;i<k;i++){
            int[] poll = priorityQueue.poll();
            res[i]=poll[0];
        }
        return res;
    }*/

/*//    338. 比特位计数
    public int[] countBits(int num) {
        int result[]=new int[num+1];
        result[0]=0;
        for(int i=1;i<=num;i++){
            if(i%2==1){
                //是奇数，那么，比前一位的偶数，多最后一位的1
                result[i]=result[i-1]+1;
            }
            else {
//               是偶数，那么最低位是0，那么/2等于右移一位，删除掉了低位的0，那么1的个数不变
                result[i]=result[i/2];
            }
        }
        return result;
    }*/
/*//    337. 打家劫舍 III     问题可以变成，父节点和子节点不能同时选中
    Map<TreeNode,Integer> f;     //取当前节点 ，那么子节点将不能取
    Map<TreeNode,Integer> g;     //不取当前节点，那么子节点可以取，也可以不取
    public int rob(TreeNode root) {
        f=new HashMap<>();
        g=new HashMap<>();
        dfs(root);
        return Math.max(g.get(root), f.get(root));
    }
    void dfs(TreeNode root){
        if(root==null){
            return ;
        }
        dfs(root.left);
        dfs(root.right);
//        取当前节点
        f.put(root, g.getOrDefault(root.left, 0)+g.getOrDefault(root.right, 0)+root.val);
//        不取当前节点
        g.put(root, Math.max(g.getOrDefault(root.left,0),f.getOrDefault(root.left, 0))+
                Math.max(g.getOrDefault(root.right,0),f.getOrDefault(root.right, 0)));

    }*/
/*//       322. 零钱兑换
//  动态规划，当前的钱，减去一个硬币的面额，那么，会有n个新的钱，新的钱中选择一个最小的数目+1，就是当前钱的最小数目
    public int coinChange(int[] coins, int amount) {
        //dp[i]组成金额数为i的时候最小的coin数
        int dp[]=new int [amount+1];
        dp[0]=0;
        for(int i=1;i<=amount;i++){
            int min=Integer.MAX_VALUE;
            for (int coin : coins) {
                //i-coin>=0说明减去这个硬币是可行的， dp[i-coin]>=0说明减去这个硬币剩下的钱，是可能组成的，不能组成的都是-1
                if(i-coin>=0&&dp[i-coin]>=0){
                    min=Math.min(min, dp[i-coin]+1);
                }
            }
            if(min==Integer.MAX_VALUE){
                dp[i]=-1;
            }
            else dp[i]=min;
        }
        return dp[amount];
    }*/
  /*  //开区间dp[i][j]最大的金币数量
//    312. 戳气球  记忆化搜索，dp[i][j]很重要，意思是：开区间（i，j）最大的硬币数。
    int dp[][];
    public int maxCoins(int[] nums) {
        if(nums.length==1){
            return nums[0];
        }
        int new_nums[]=new int[nums.length+2];
        new_nums[0]=1;
        new_nums[new_nums.length-1]=1;
        for (int i = 0; i < nums.length; i++) {
            new_nums[i+1]=nums[i];
        }
        dp=new int[new_nums.length][new_nums.length];
        for (int i = 0; i < dp.length; i++) {
            for (int i1 = 0; i1 < dp[i].length; i1++) {
                dp[i][i1]=-1;
            }
        }
//        搜索的是数组，开区间（i，j）内最大硬币数，因为获取开区间的最大硬币数，需要遍历，最后一个戳破的气球k，那么戳破气球k需要，开区间的边界nums[i]
//        nums[j]的值，所以设置一个新的数组，两边都为1，这样就不用再进行考虑是不是边界i，j，越界了
        int search = search(0, new_nums.length-1, new_nums);
        return  search;
    }

    int search (int i,int j,int[] nums){
//        已经搜索过的情况
        if(dp[i][j]!=-1){
            return dp[i][j];
        }
        if(j-i==1){
            dp[i][j]=0;
            return dp[i][j];
        }
        else {
            int max=0;
            for(int k=i+1;k<j;k++){
//                开区间i，j中最后一个被戳破的是k，那么

                int i1 = search(i, k, nums) + nums[i] * nums[k] * nums[j] + search(k, j, nums);
                max=Math.max(i1, max);
            }
            dp[i][j]=max;
            return dp[i][j];
        }
    }*/
/*//    309. 最佳买卖股票时机含冷冻期
    public int maxProfit(int[] prices) {
//        前一天，dp1持有股票 dp2不持有股票即将进入冷冻期（也就是说卖出股票） dp3不持有股票不进入冷冻期 的最大值
        int dp1=-prices[0],dp2=0,dp3=0;
        int dp1cur=-prices[0],dp2cur=0,dp3cur=0;
        for(int i=1;i<prices.length;i++){
            //两种操作，一种是买入，一种是卖出
            dp2cur=dp1+prices[i];
            dp1cur=Math.max(dp1,dp3-prices[i]);
            dp3cur=Math.max(dp2,dp3);

            dp1=dp1cur;
            dp2=dp2cur;
            dp3=dp3cur;
        }
        return Math.max(dp2, dp3);
    }*/

//    301. 删除无效的括号
/**
 *  关键是想到怎么算出来,需要减少的左右括号数量,算出来之后,通过回溯算法,尝试删除,还是保留每一个数字,但是要剪枝,设个leftcount,rightcount,left>right才能选择右括号
 *  leftremove>0才能移除左括号
 */

   /* HashSet<String> set;
    char [] array;
    int len;
    public List<String> removeInvalidParentheses(String s) {
        int left=0,right=0;
        set=new HashSet<>();
        array = s.toCharArray();
        len=array.length;
        for(char c :s.toCharArray()){
            if(c=='('){
                left++;
            }
            else if(c==')'){
                if(left!=0){
                    left--;
                }
                else {
                    right++;
                }
            }
        }
        backtrack(0, 0, left, right, 0,new StringBuffer());
        return new ArrayList<String>(set);
    }
    void backtrack(int leftCount,int rightCount,int leftRemove,int rightRemove,int index,StringBuffer path){
//   结束位置
        if(index==len){
            if(leftRemove==0&&rightRemove==0) {
                set.add(path.toString());
            }
//                既然要结束递归得有return啊 不管放不放入结果集,都要结束递归了
            return;
        }

//        有两种情况,一个是删除当前字符,一个是保留当前字符,两个都需要递归尝试一下

//      1删除当前字符
        if(array[index]=='('&&leftRemove>0){
            backtrack(leftCount, rightCount, leftRemove-1, rightRemove, index+1, path);
        }
        else if(array[index]==')'&&rightRemove>0){
            backtrack(leftCount, rightCount, leftRemove, rightRemove-1, index+1, path);
        }


//        2保留当前字符
        path.append(array[index]);
        if(array[index]!='('&&array[index]!=')'){
            backtrack(leftCount, rightCount, leftRemove, rightRemove, index+1, path);
        }
        else if(array[index]=='('){
            backtrack(leftCount+1, rightCount, leftRemove, rightRemove, index+1, path);
        }
        else if(leftCount>rightCount){
            backtrack(leftCount, rightCount+1,leftRemove,rightRemove,index+1,path);
        }
//        遍历完成之后,给加入的删除了,让上级函数,调用继续回溯函数时的path相同
        path.deleteCharAt(path.length()-1);
    }*/
/*//    300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
//        这个list，每个位置i都代表长度为i+1的序列最后一位的最小可以取到的值
        List<Integer> list=new ArrayList<>();
        list.add(nums[0]);
        for(int i=1;i<nums.length;i++){
            //如果遍历到的这个数，比当前最长序列的最后一位大，那么序列可以增长一位
            if(nums[i]>list.get(list.size()-1)){
                list.add(nums[i]);
            }
//            否则，看看可不可以让某一长度的最后一位的最小值变小，本质是贪心算法
            else {
                int i1 = Collections.binarySearch(list, nums[i]);
                if(i1<0){
                    i1++;
                    i1=Math.abs(i1);
                    list.remove(i1);
                    list.add(i1, nums[i]);
                }
            }
        }
        return list.size();
    }*/
/**  279. 完全平方数
 * 一个numSquares（n）等于numSquares（n-k*k）+1，k*k算一个完全平方数了，然后，就往前面找，前面（n-k*k）的
 * 结果已经算出来了，所以只要枚举k就行了，找到那个最小的值，就是正确的答案了
 * @param n
 * @return
 */
//public int numSquares(int n) {
//        int dp[]=new int[n+1];
//        dp[0]=0;
//        for(int i=1;i<=n;i++){
//        int k=1;
//        int ans=Integer.MAX_VALUE;
//        while (i-k*k>=0){
//        ans=Math.min(ans, dp[i-k*k]+1);
//        k++;
//        }
//        dp[i]=ans;
//        }
//        return dp[n];
//        }
/**
 *         215. 数组中的第K个最大元素
 *         由这个题，写了一个堆，实现了这个数据结构，之前一直了解，但是自己动手实现过
 */
/*class  Heap{
    int []tree;
    int capacity;
    int size=0;
    public Heap(int capacity){
        this.tree=new int[capacity+1];
        this.size=0;
    }
    public void createHeap(int []a){
        for(int i=0;i<a.length;i++){
            tree[i+1]=a[i];
        }
        this.size=a.length;
        int i=size/2;
        while (i!=0){
            shiftDown(i);
            i--;
        }
    }
    public void add(int x){
        size++;
        tree[size]=x;
        shiftUp(size);
    }
    public int  pop(){
        int pop=tree[1];
        tree[1]=tree[size];
        tree[size]=0;
        size--;
        shiftDown(1);
        return pop;
    }
    //主要的两个方法，向上，向下调整
    void shiftDown(int i){
        int val=tree[i];
        while (i*2<=size){
            if(i*2+1<=size){
                int val1=tree[i*2];
                int val2=tree[i*2+1];
                if(val1>val2){
                    if(val1>val){
                        swap(tree, i, i*2);
                        i*=2;
                    }
                    else {
                        break;
                    }
                }
                else {
                    if(val2>val){
                        swap(tree, i, i*2+1);
                        i=i*2+1;
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                int val1=tree[2*i];
                if(val1>val){
                    swap(tree, i, i*2);
                    i*=2;
                }
                else {
                    break;
                }
            }
        }
        tree[i]=val;
    }
    void shiftUp(int i){
        int val=tree[i];
        while (i!=0){
            if (i==1){
                break;
            }
            if(val>tree[i/2]){
                swap(tree, i, i/2);
            }
            else {
                break;
            }
            i/=2;
        }
        tree[i]=val;
    }
    public void  swap(int nums[],int i,int j){
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
}*/
/**
 * 215. 数组中的第K个最大元素
 * 快速排序改良版，可以实现线性复杂度的选择算法
 * 快速排序，引入随机挑选，分割的位置，并且，每次分隔完成后，返回分隔位置，因为这个分割位置已经定了，已经是排序完成后的位置了
 * 作比较，如果是想要的位置，就直接返回不需要再排序了。
 *
 *
 * 分割细节点，找到一个随机位置以后，先把这个位置和最右边位置交换，然后，遍历【l，r-1】的位置，进行比较，
 * 设置一个变量i，i表示i以前的位置都已经是小于等于x的了，
 * 遍历完成之后，再给i和r换位置，给之前那个数换回来。
 */
class Solution {

    Random random=new Random();
    public int findKthLargest(int[] nums, int k) {
        int l=0;
        int r=nums.length-1;
        int index=nums.length-k;
        return selectMaxK(nums, l, r, index);
    }
    public int selectMaxK(int []nums,int l,int r,int index){
        int q=randomPartition(nums,l,r);
        if(q==index){
            return nums[q];
        }
        else {
            if(q<index){
                return selectMaxK(nums, q+1, r, index);
            }
            else {
                return selectMaxK(nums, l, q-1, index);
            }
        }
    }
    public int randomPartition(int []nums,int l,int r){
        int i = random.nextInt(r - l + 1) + l;
        int x=nums[i];
        //给选定的随机位置，换到最右边，然后分好位置后，再给他换回来
        swap(nums, i, r);
        i=l;
        for(int j=l;j<=r-1;j++){
            if(nums[j]<=x){
                swap(nums, i++, j);
            }
        }
        swap(nums, i, r);
        return i;
    }


    public void  swap(int nums[],int i,int j){
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
}
public class Finished {
    //    581. 最短无序连续子数组
//    O(n)的时间复杂度，空间复杂度为1，
//    只需要算出逆序对的最小值，和逆序对的最大值，然后找到他们在顺序排列的位置，就行了
//    从左到右，找到第一个逆序对，然后开始算最小值，从右到左找到第一个逆序对，开始算最大值
    public int findUnsortedSubarray(int[] nums) {
        boolean flag1=true;
        int max=Integer.MIN_VALUE,min=Integer.MAX_VALUE;
        for(int i=0;i<nums.length-1;i++){
            if(flag1){
                if(nums[i]>nums[i+1]){
                    min=nums[i+1];
                    flag1=false;
                }
            }
            else {
                min=Math.min(min, nums[i+1]);
            }
        }
        boolean flag2=true;
        for(int i=nums.length-1;i>0;i--){
            if(flag2){
                if(nums[i]<nums[i-1]){
                    max=nums[i-1];
                    flag2=false;
                }
            }
            else {
                max=Math.max(max, nums[i-1]);
            }
        }
        int start=0,end=0;
        if(flag1){
            return 0;
        }
        else {
            for(int i=0;i<nums.length;i++){
                if(nums[i]>min){
                    start=i;
                    break;
                }
            }
            for (int i=nums.length-1;i>=0;i--){
                if(nums[i]<max){
                    end=i;
                    break;
                }
            }
            return end-start+1;
        }
    }
    //    448. 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {

        List<Integer> ans  = new ArrayList<>();
        int n=nums.length;
        for(int i=0;i<n;i++){
//            说明有x，那么给第x个数，增加一个n+1，作为标记，说明x存在了
            int x=nums[i]%(n+1);
//            第x个数是nums[x-1]
            nums[x-1]=nums[x-1]+n+1;
        }
        for(int i=0;i<n;i++){
            if(nums[i]<=n){
                ans.add(i+1);
            }
        }
        return ans;
    }
    /**
     *     538. 把二叉搜索树转换为累加树
     *     二叉搜索树中序遍历是从小到大的，那么逆序遍历就是从大到小的，然后你序遍历，存下每个逆序遍历的和，顺便更新每个节点

     int sum=0;
     public TreeNode convertBST(TreeNode root) {
     dfs(root);
     return root;
     }
     private void dfs(TreeNode node){
     if(node==null) return;
     dfs(node.right);
     sum+=node.val;
     node.val=sum;
     dfs(node.left);
     }*/

    /*//    494. 目标和
    public int findTargetSumWays(int[] nums, int target) {
//        dp[i][j]为，前i个数，通过+-组成j的排列数。 数组下标不能为负数，所以，设，实际的数要＋1000;
        int dp[][]=new int[nums.length][2001];
        dp[0][nums[0]+1000]=1;
        dp[0][-nums[0]+1000]=1;
        if(nums[0]==0){
            dp[0][1000]=2;
        }
        for(int i=1;i<nums.length;i++){
            for(int j = -1000;j<=1000;j++){
                if(j-nums[i]+1000<0){
                    dp[i][j+1000]=dp[i-1][j+nums[i]+1000];
                }
                else if(j + nums[i] + 1000>2000){
                    dp[i][j + 1000] = dp[i - 1][j - nums[i] + 1000];
                }
                else {
                    dp[i][j + 1000] = dp[i - 1][j + nums[i] + 1000] + dp[i - 1][j - nums[i] + 1000];
                }
            }
        }
        return dp[nums.length-1][target+1000];
    }*/
    /*     437. 路径总和 III
//    key is the prefixSum,value is count of the node which prefixSum is key
    private Map<Integer,Integer> map=new HashMap<>();
    int target;
    public int pathSum(TreeNode root, int targetSum) {
        target=targetSum;
//        空树，prefixSum=0, key =1
        map.put(0, 1);
        return  dfs(root,  0);
    }
    int dfs(TreeNode node,int prefixSum){
        if (node==null)return 0;
        prefixSum+=node.val;
        //获得终点为node的路径，且路径的和为target，获得的那个节点其实是，路径起始节点的父节点，这也是为什么，要设置{0：1}的原因
        int res=map.getOrDefault(prefixSum-target,0);
        map.put(prefixSum, map.getOrDefault(prefixSum, 0)+1);
        int left= dfs(node.left, prefixSum);
        int right=dfs(node.right,prefixSum);
//        计算完这个节点，以及其子节点，应该删除，避免他的兄弟节点是子节点，向上查询的时候出现问题，误认为他也是父节点，可以组成路径
        map.put(prefixSum, map.get(prefixSum)-1);
        return res + left + right;
    }*/
    /*         416. 分割等和子集
    public boolean canPartition(int[] nums) {
        if(nums.length<2)return false;
        int sum=0;
        for (int num : nums) {
            sum+=num;
        }
        if(sum%2==1){
            return false;
        }
        int target=sum/2;
        int n=nums.length;
        //dp[i][j] 前[0,i]个数字，能否组成和为j
        boolean dp[][]=new boolean[n][target+1];
        if(nums[0]<target){
            dp[0][nums[0]]=true;
        }
//        如果值为零，前i个数都能凑齐
        for(int i=0;i<n;i++){
            dp[i][0]=true;
        }
        for(int i=1;i<n;i++){
            for(int j=1;j<=target;j++){
                if(nums[i]>j){
                    dp[i][j]=dp[i-1][j];
                }
                else {
                    dp[i][j]=dp[i-1][j]||dp[i-1][j-nums[i]];
                }
            }
        }
        return dp[n-1][target];
    }*/
    /*236. 二叉树的最近公共祖先 后序遍历，想出什么情况该节点是所需要的节点*/
    /*TreeNode ans;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs(root, p, q);
        return ans;
    }
    public boolean dfs(TreeNode root,TreeNode p ,TreeNode q){
        if(root==null) return false;
        boolean leftChid=dfs(root.left,p,q);
        boolean rightChid=dfs(root.right,p,q);
        //判断该节点是否是最近公共祖先节点，要么，p,q分别在左右子树中，要么该节点就是p或者q，另一个节点在他的子数中。
        if(  (leftChid&&rightChid)  ||  (  (root.val==p.val||root.val==q.val)&&(rightChid||leftChid)  )   ) {
            ans=root;
        }
        return leftChid||rightChid||root.val==p.val||root.val==q.val;
    }*/
    /**
     * 221. 最大正方形
     * dp[i][j]以i，j为右下角的最大边长
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int dp[][] = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(i==0||j==0){
                    if(matrix[i][j]=='1'){
                        dp[i][j]=1;
                    }
                }
                else {
                    if(matrix[i][j]=='1'){
                        dp[i][j]=Math.min(Math.min(dp[i-1][j],dp[i][j-1]) ,dp[i-1][j-1])+1;
                    }
                }
            }
        }
        int max=0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                max=Math.max(dp[i][j],max);
            }
        }
        return max*max;
    }
    /*208. 实现 Trie (前缀树)
class Trie {
    private Trie[] children;
    private boolean isEnd;
    *//** Initialize your data structure here. *//*
    public Trie() {
        children=new Trie[26];
        isEnd=false;
    }

    *//** Inserts a word into the trie. *//*
    public void insert(String word) {
        Trie trie=this;
        for(char c:word.toCharArray()) {
            if(trie.children[c-'a']==null){
                trie.children[c-'a']=new Trie();
            }
            trie=trie.children[c-'a'];
        }
        trie.isEnd=true;
    }

    *//** Returns if the word is in the trie. *//*
    public boolean search(String word) {
        Trie trie=this;
        for(char c:word.toCharArray()) {
            if(trie.children[c-'a']!=null){
                trie=trie.children[c-'a'];
            }
            else {
                return false;
            }
        }
        if(trie.isEnd){
            return true;
        }
        else return false;

    }

    *//** Returns if there is any word in the trie that starts with the given prefix. *//*
    public boolean startsWith(String prefix) {
        Trie trie=this;
        for(char c:prefix.toCharArray()) {
            if(trie.children[c-'a']!=null){
                trie=trie.children[c-'a'];
            }
            else {
                return false;
            }
        }
        return true;

    }
}*/
    //207. 课程表,简单的拓扑排序，不过需要给他给的图转换成常规的图
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //indegree为0的时候，就可以进入了
        int[] indegree =new int [numCourses];
        Map<Integer,List<Integer>> G=new HashMap<>();
        //转换图为节点，加指向的链表的形式
        for(int i=0;i<prerequisites.length;i++){
            indegree[prerequisites[i][0]]++;
            if(G.containsKey(prerequisites[i][1])){
                G.get(prerequisites[i][1]).add(prerequisites[i][0]);
            }
            else {
                ArrayList<Integer> list = new ArrayList<>();
                list.add(prerequisites[i][0]);
                G.put(prerequisites[i][1],list);
            }
        }
        Queue<Integer> queue=new LinkedList<>();
        //把可以遍历的放入其中
        for(int i=0;i<indegree.length;i++){
            if(indegree[i]==0){
                queue.add(i);
            }
        }
        int count=0;
        while (!queue.isEmpty()){
            count++;
            //访问一个点
            Integer poll = queue.poll();
            //遍历他的度
            List<Integer> list = G.get(poll);
            if(list!=null){
                for(int i=0;i<list.size();i++){
                    indegree[list.get(i)]--;
                    if(indegree[list.get(i)]==0){
                        queue.add(list.get(i));
                    }
                }
            }
        }
        if(count==numCourses){
            return true;
        }
        else return false;
    }

    /*//229. 求众数 II
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> list = new ArrayList<>();
        int cond1=nums[0],cond2=nums[0];
        int count1=0,count2=0;
        for (int num : nums) {
            if(cond1==num){
                count1++;
                continue;
            }
            if(cond2==num){
                count2++;
                continue;
            }
            if(count1==0){
                cond1=num;
                count1++;
                continue;
            }
            if(count2==0){
                cond2=num;
                count2++;
                continue;
            }
            count1--;
            count2--;
        }
        if(n(nums,cond1)>nums.length/3){
            list.add(cond1);
        }
        if(n(nums,cond2)>nums.length/3){
            list.add(cond2);
        }
        return list;
    }

    int n(int [] nums,int x){
        int count=0;
        for(int i=0;i<nums.length;i++){
            if(x==nums[i]){
                count++;
            }
        }
        return count;
    }*/
    //找多数元素一个数，如果相同就count++，否则就--，如果到了0，就换下一个数，这样最后留下来的数
   // 肯定是出现次数最多的数  时间复杂度是n，空间复杂度是1
    public int majorityElement(int[] nums) {
        int condNums=nums[0];
        int count=1;
        for(int i=1;i<nums.length;i++){
            if(nums[i]==condNums){
                count++;
            }
            else{
                count--;
                if(count==0){
                    condNums=nums[i];
                    count=1;
                }
            }
        }
        return count;
    }
    /* //    160. 相交链表
//    如果相交，那么从从后往前数若干个是相同的，就从后往前截断，让两个链表的长度相同，然后同时向后遍历
    //如果有节点==那么就相交了
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenthA=0,lenthB=0;
        ListNode cur=headA;
        while (cur!=null){
            lenthA++;
            cur=cur.next;
        }
        cur=headB;
        while (cur!=null){
            lenthB++;
            cur=cur.next;
        }
        ListNode curA=headA;
        ListNode curB=headB;
        if(lenthA>lenthB){
            int sub = lenthA - lenthB;
            for(int i=0;i<sub;i++){
                curA=curA.next;
            }
            while (curA!=null&&curB!=null){
                if(curA==curB){
                    return curA;
                }
                curA=curA.next;
                curB=curB.next;
            }
        }
        else {
            int sub = lenthB - lenthA;
            for(int i=0;i<sub;i++){
                curB=curB.next;
            }
            while (curA!=null&&curB!=null){
                if(curA==curB){
                    return curA;
                }
                curA=curA.next;
                curB=curB.next;
            }

        }
        return null;

    }*/

    /*//**
     * 148.链表的归并排序  递归版本
     * @param head
     * @return
     */
   /* public ListNode sortList(ListNode head) {
        if(head==null||head.next==null){
            return head;
        }
        ListNode slow=head;
        ListNode fast=head.next;
        //快慢指针的循环条件，快指针如果不能走，那么慢指针也不要走
        while (fast!=null&&fast.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        ListNode right=slow.next;
        slow.next=null;
        ListNode left=sortList(head);
        right=sortList(right);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left!=null&&right!=null){
            if(left.val<right.val){
                h.next=left;
                left=left.next;
            }
            else {
                h.next=right;
                right=right.next;
            }
            h=h.next;
        }
        h.next=(right!=null)?right:left;
        return res.next;
    }*/

        /**
         * 146. LRU 缓存机制
         * 自己写一个双向链表，保存好伪头和伪尾，新进来的数据，使用过的数据，都放在头，并且删除原来的数据，删除数据只需要o（1）
         * 是因为，map中存储的是链表节点对象，把这个节点删除用的是o（1）。
         */
        class LRUCache {
            class DLinkNode{
                int key;
                int value;
                DLinkNode pre;
                DLinkNode next;
                public DLinkNode(){

                };
                public DLinkNode(int key,int value){
                    this.key=key;
                    this.value=value;
                }
            }
            DLinkNode head=new DLinkNode();
            DLinkNode tail=new DLinkNode();
            int size=0;
            private int capacity;
            private Map<Integer, DLinkNode> map = new HashMap<>();

            public LRUCache(int capacity) {
                this.capacity = capacity;
                head.next=tail;
                tail.pre=head;
            }

            public int get(int key) {
                if(map.keySet().contains(key)){
                    DLinkNode dLinkNode = map.get(key);
                    remove(dLinkNode);
                    addHead(dLinkNode);
                    return dLinkNode.value;
                }
                else {
                    return -1;
                }
            }

            private void remove(DLinkNode dLinkNode) {
                dLinkNode.pre.next=dLinkNode.next;
                dLinkNode.next.pre=dLinkNode.pre;
            }
            private void addHead(DLinkNode dLinkNode) {
                dLinkNode.next=head.next;
                dLinkNode.pre=head;
                head.next.pre=dLinkNode;
                head.next=dLinkNode;
            }

            public void put(int key, int value) {
                if(map.keySet().contains(key)){
                    DLinkNode dLinkNode = map.get(key);
                    dLinkNode.value=value;
                    remove(dLinkNode);
                    addHead(dLinkNode);
                }
                else if(size<capacity){
                    DLinkNode dLinkNode = new DLinkNode(key, value);
                    map.put(key, dLinkNode);
                    addHead(dLinkNode);
                    size++;
                }
                else {
                    int k = removeTail();
                    map.remove(k);
                    DLinkNode dLinkNode = new DLinkNode(key, value);
                    map.put(key, dLinkNode);
                    addHead(dLinkNode);
                }
            }

            private int removeTail() {
                DLinkNode pre = tail.pre;
                remove(pre);
                return pre.key;
            }
        }


        /**    142. 环形链表 II
         * 笨办法，都存入Hashset中，如果碰到相同的说明有环，并且此处是入环点，时间，空间都是N
         * 快慢双指针，弗洛伊德判环法；
         * 双指针判环，进环前为a，p1走过的距离的环为b，剩下的环为c，根据距离两倍可以得到关系a=c，所以从
         * p1，p2接触点跳到环入口和从head跳到入口距离相等
         * 2(a+b)=a+b+c+b
         * a=c
         */
        public Main.ListNode detectCycle(Main.ListNode head) {
            Main.ListNode ans = null;
            if (head == null) return ans;
            Main.ListNode p1 = head;
            Main.ListNode p2 = head.next;
            Main.ListNode p3 = null;
            while (null != p2) {
                if (p2 == p1) {
                    p3 = p1;
                    break;
                } else {
                    p1 = p1.next;
                    p2 = p2.next;
                    if (p2 != null) {
                        p2 = p2.next;
                    }
                }
            }
            if (p3 == null) {
                return ans;
            } else {
                //因为从head前一位置开始算a了，=head就等于先跳了一下了，所以相遇位置，也得跳一下
                p3 = p3.next;
                while (head != p3) {
                    p3 = p3.next;
                    head = head.next;
                }
                return head;
            }
        }
    /*139. 单词拆分
     *   动态规划，判断前i个字符是否符合，就分成s1和s2（包含第i个）两部分，设定一个点，j来分割，j的范围为，j到i小于等于单词的最大值，
     * 这用于剪枝，如果有符合的，就dp【i】为真了，让后面的继续使用。
     * 关键在于推dp的时候竟然多个变量j，没有考虑到。
     *
     * */
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set=new HashSet<>();
        int maxlen=0;
        for(String string:wordDict){
            if(string.length()>maxlen){
                maxlen=string.length();
            }
            set.add(string);
        }
        boolean dp[]=new boolean[s.length()];
        char[] str = s.toCharArray();
        for(int i=0;i<str.length;i++){
            for(int j=i;j>=0&&i-j+1<=maxlen;j--){
                String s1 = new String(str, j, i - j + 1);
                if(set.contains(s1)){
                    if(j==0){
                        dp[i]=true;
                        break;
                    }
                    else {
                        if(dp[j-1]==true) {
                            dp[i]=true;
                            break;
                        }
                    }
                }
                else {
                    continue;
                }
            }
        }
        return dp[str.length-1];

    }
    /*128. 最长连续序列
     *    数字存入hashset中，空间换时间，找，i+1是否存在，只需要用O（1）时间复杂度，并且从连续数字的第一哥树开始找，
     * 所以复杂度是O（n），
     *
     * */
    public int longestConsecutive(int[] nums) {
        Set<Integer> set=new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int count=0,maxConut=0;
        for (Integer integer : set) {
            count=0;
            //避免重复，所以如果存在integer-1，那就没必要从integer开始寻找了
            if(set.contains(integer-1)){
                continue;
            }
            else {
                count++;
                integer=integer+1;
                while (set.contains(integer)){
                    integer++;
                    count++;
                }
                maxConut=Math.max(count, maxConut);
            }
        }
        return maxConut;
    }

    //124. 二叉树中的最大路径和
    //得到一颗树，root自己作为连接点，连接左右子树的路径和，与max比较，并返回自己如果作为与上面连接的时候最大的贡献值
    int max=Integer.MIN_VALUE;
    public int maxPathSum(Main.TreeNode root) {
        gainMax(root);
        return max;
    }
    int gainMax(Main.TreeNode root){
        if(root==null)return 0;
        if(root.left==null&&root.right==null){
            //得到路径和
            if(root.val>max) max=root.val;
            //返回贡献值
            return Math.max(0, root.val);
        }
        int left = gainMax(root.left);
        int right= gainMax(root.right);
        //这个节点的路径和
        if (root.val+left+right>max) max=root.val+left+right;
//        这个节点的贡献值
        if(root.val+Math.max(right,left)>0){
            return Math.max(right,left)+root.val;
        }
        else {
            return 0;
        }
    }
    //原地算法，114.二叉树展开成链表，给这个树，左子树插入到根的右子树位置，右子树放到左子树的右侧。
    public void flatten(Main.TreeNode root) {
        if(root==null) return;
        while (root!=null){
            if(root.left==null){
                root=root.right;
            }
            else {
                Main.TreeNode left=root.left;
                Main.TreeNode right=root.right;
                root.left=null;
                root.right=left;

                while (left.right!=null){
                    left=left.right;
                }
                left.right=right;


            }
        }
    }
    /*
    101.对称二叉树
    一个树是不是对称二叉树，要看他的子树是不是对称
    子树对称需要满足：1，两个树的根相同，2，两个树的左子树和另一个树的右子树对称
     */
    public boolean isSymmetric(Main.TreeNode root) {
        return check(root.left, root.right);
    }
    boolean check(Main.TreeNode root1, Main.TreeNode root2){
        if(root1==null&&root2==null){
            return true;
        }
        if(root1==null||root2==null){
            return false;
        }
        if(root1.val!=root2.val){
            return false;
        }
        return check(root1.left, root2.right)&&check(root1.right, root2.left);
    }
    /* 85.最大矩形
       单调栈，和上一个题优化方法差不多
    */
    public int maximalRectangle(char[][] matrix) {
        if(matrix.length==0)return 0;
        int m=matrix.length,n=matrix[0].length;
//        把找矩形的问题转化成一列一列的柱形，找柱形中最大的矩形
        //left数组代表从左往右数，到这个位置的最多1的数量，然后left的一列，可以看做是一个柱形，找到最大的矩形就行了
        //所以用单调栈
        int left[][]=new int[m][n];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j]=='1'){
                    left[i][j]=(j==0?0:left[i][j-1])+1;
                }
            }
        }
        int max=0;
        for(int j=0;j<n;j++){
            Deque<Integer> stack=new LinkedList<>();
            for(int i=0;i<m;i++){
                while(!stack.isEmpty()&&left[i][j]<left[stack.peek()][j]){
                    Integer pop = stack.pop();
                    while (!stack.isEmpty()&&left[stack.peek()][j]==left[pop][j]) {
                        pop = stack.pop();
                    }
                    if(!stack.isEmpty()){
                        int x=(i-stack.peek()-1)*left[pop][j];
                        max=Math.max(x,max);
                    }
                    else {
                        int x=i*left[pop][j];
                        max=Math.max(x,max);
                    }
                }
                stack.push(i);
            }
            while (!stack.isEmpty()){
                int pop=stack.pop();
                if(!stack.isEmpty()){
                    int x=(m-stack.peek()-1)*left[pop][j];
                    max=Math.max(x,max);
                }
                else {
                    int x=m*left[pop][j];
                    max=Math.max(x,max);
                }
            }
        }
        return max;
    }
    /*
    84. 柱状图中最大的矩形
    单调栈，单调栈，就是从左到右，逻辑元素以此增大的栈
    每次进入的元素，都干掉了前面的比他大的元素，保留了比他小的元素
 */
    public int largestRectangleArea(int[] heights) {
        if (heights.length == 0) return 0;
        if (heights.length == 1) return heights[0];
        int max = 0;
//        栈
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < heights.length; i++) {
//            判断栈如果不空，并且栈顶元素比要进入的元素大，那么就干掉栈顶元素
            while ((!stack.isEmpty()) && (heights[stack.peek()] > heights[i])) {
//                干掉了栈顶元素
                Integer pop = stack.pop();
//                判断栈顶元素和前一个栈顶元素是不是相等，做循环，如果都相同，得给他们都干掉
                while (!stack.isEmpty()&&heights[stack.peek()]==heights[pop]) {
                    pop=stack.pop();
                }
//                  开始算面积，从要入队的元素，到前面比栈顶元素小的元素之间的元素，因为栈顶元素已经给比他大的元素干掉了
//              所以找到前面的比他小的元素的后一个到如对元素的前一个都是合格的面积
                if(!stack.isEmpty()){
                    int x = (i - stack.peek()-1) * heights[pop];
                    max = Math.max(x, max);
                }
//                没有前面的元素，也就是从零开始，到i都是合格的
                else {
                    int x = (i) * heights[pop];
                    max = Math.max(x, max);
                }
            }
//            前面已经处理好比他大的元素了，新元素入队
            stack.push(i);
        }
//          后面肯定比他大了因为没干掉他，前面直到前一个元素之前都比他大，因为他干掉了他们。
        while (stack.size() > 1) {
            Integer pop = stack.pop();
            int x = (heights.length - stack.peek()-1) * heights[pop];
            max = Math.max(x, max);
        }
//        前面都被干掉了，说明他比前面都小，后面干不掉他，说明他比后面都小
        if (stack.size() == 1) {
            Integer pop = stack.pop();
            int x = (heights.length) * heights[pop];
            max = Math.max(x, max);
        }
        return max;
    }
    /*
    78. 子集
    组合，不是排列，不要做一个排列，然后从中挑选出组合，这样复杂度太高了
    组合，从上而下搜索，所以需要一个标记int，确定前面的都被搜索过了，他只需要遍历后面的数字就行了，前面的数字有其他递归去遍历
    就是1开头的，只需要考虑后面的，2345，2开头的，就不要考虑1了，只考虑后面的就行了
    临时list，放一个数字以后，然后开始递归，将他后面的数字都递归一遍，等回来的时候，再给这个数字删除，继续递归下一个数字。

     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        backTrack(0,ans, nums, new ArrayList<Integer>());
        return ans;
    }

    public void backTrack(int i,List<List<Integer>> ans, int[] nums, List<Integer> list) {
        ans.add(new ArrayList<>(list));
        for(int j=i;j<nums.length;j++){
            list.add(nums[j]);
            backTrack(j+1, ans, nums, list);
            list.remove(list.size()-1);
        }
    }


    /*
49. 字母异位词分组
主要是hashmap的用法，map自身也可以做Map的key。如果一个map完全一样那么他们的就相等，eque相等，hashcode也相等。
 */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<Map<Character,Integer>,ArrayList<String>> maps=new HashMap<>();
        List<List<String>> res=new ArrayList<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Map<Character,Integer> map=new HashMap<>();
            for (char aChar : chars) {
                if(map.containsKey(aChar)){
                    map.put(aChar, map.get(aChar)+1);
                }
                else {
                    map.put(aChar, 1);
                }
            }
            if(maps.containsKey(map)){
                maps.get(map).add(str);
            }
            else {
                ArrayList<String> strings = new ArrayList<>();
                strings.add(str);
                maps.put(map, strings);
            }
        }
        for(ArrayList<String> strings:maps.values()){
            res.add(strings);
        }
        return res;
    }
    /*
    48. 旋转图像
        要的是原地旋转，所以不能再创建一个矩阵数组，把他写到新的数组中
        旋转90°其实就是先水平反转，再对角线反转，这样就可以了，时间复杂度都是N2,空间复杂度为1
     */
    public void rotate(int[][] matrix) {
        int temp;
        for(int i=0;i<matrix.length/2;i++){
            for(int j=0;j<matrix[i].length;j++){
                temp=matrix[i][j];
                matrix[i][j]=matrix[matrix.length-i-1][j];
                matrix[matrix.length-i-1][j]=temp;
            }
        }
        for(int i=0;i<matrix.length;i++){
            for(int j=0;j<i;j++){
                temp=matrix[i][j];
                matrix[i][j]=matrix[j][i];
                matrix[j][i]=temp;
            }
        }

    }


    //42. 接雨水
    /*
    接雨水关键在于理解，怎么计算每个位置接到的雨水，每个位置接到的雨水等于，左边和右边的高度最大值中的最小值，减去自己的高度。
     */
    public int trap(int[] height) {
        if(height.length<3)return 0;
        int maxleft[]=new int[height.length];
        int maxright[]=new int[height.length];
        maxleft[0]=height[0];
        maxright[height.length-1]=height[height.length-1];
        for(int i=1;i<height.length;i++){
            maxleft[i]=Math.max(maxleft[i-1],height[i]);
        }
        for(int i=height.length-2;i>0;i--){
            maxright[i]=Math.max(maxright[i+1],height[i]);
        }
        int res=0;
        for(int i=1;i<height.length-1;i++){
            int min=Math.min(maxleft[i],maxright[i]);
            if(min-height[i]<0){

            }
            else {
                res+=min-height[i];
            }
        }
        return res;

    }
    //    34. 在排序数组中查找元素的第一个和最后一个位置
    /*
    用二分找到，大于等于目标值的数
    找到大于目标值的数，就行了
     */
//    public int[] searchRange(int[] nums, int target) {
//        if(nums.length==1){
//            if(nums[0]==target) return new int []{0,0};
//            else{
//                return new int[]{-1,-1};
//            }
//        }
//
//        int x1=binarySearch(nums, target, true);
//        int x2=binarySearch(nums, target, false);
//        if(x1!=nums.length&&nums[x1]==target&&nums[x2-1]==target){
//            return new int[]{x1,x2-1};
//        }
//        else {
//            return  new int[] {-1,-1};
//        }
//    }
//    //找到第一个大于或者大于等于target的数,根据tager
//    public int binarySearch(int []nums,int target,boolean flag){
//        int left=0;
//        //用来记录可能的值，然后不断二分找到最可能的值
//        int ans=nums.length;
//        int right=nums.length-1;
//        while (left<=right){
//            int mid=(left+right)/2;
//            if(nums[mid]>target||(nums[mid]>=target&&flag)){
//                ans=mid;
//                right=mid-1;
//            }
//            else {
//                left=mid+1;
//            }
//        }
//        return ans;
//    }
    //    33. 搜索旋转排序数组
    /*
    二分查找，因为只有一个反转，所以可以多做一次判断nums[l]<=nums[mid]，来确定是那一部分是有序的，
    然后根据有序的部分，判断target在不在有序的范围内，确定l和r，进行二分。
     */
//    public int search(int[] nums, int target) {
//        int n = nums.length;
//        int l=0,r=n-1;
//        while (l<=r){
//            int mid=(l+r)/2;
//            if(nums[mid]==target){
//                return mid;
//            }
//            else{
//                if(nums[l]<=nums[mid]){
//                    if(nums[l]<=target&&target<nums[mid]){
//                        r=mid-1;
//                    }
//                    else {
//                        l = mid + 1;
//                    }
//                }
//                else {
//                    if(nums[mid]<target&&target<=nums[r]){
//                        l=mid+1;
//                    }
//                    else {
//                        r=mid-1;
//                    }
//                }
//            }
//        }
//        return -1;
//    }
    //    32. 最长有效括号
    /*
     *  暴力方法，枚举所有的字符串，然后判断是否的正确的，需要O（n3）的复杂度，原因是进行了很多重复的判断
     * 相邻增加一个两个字符和前面是否能组成有效括号，直接根据前面的结果来得到后面的结果，所以可以用动态规划
     * 要得到最长的有效括号，dp[i]为结尾的长度，那么如果chars[i]是'('就为0，不可能以‘（‘结尾的合法字符串，
     * 如果是’）‘解为，那么可以看chars【i-1】是不是’（‘，来和dp【i】组成一个新的括号，如果可以，那么dp【i】=dp【i-2】+2
     * 如果不可以，那么还得判断是不是’（（。。。））‘这样的情况，所以，可以根据dp【i-1】的值，查到他合法的第一位字符串的前面
     * 是什么。那个位置是chars【i-dp【i-1】-1】如果这个chars【i-dp【i-1】-1】='('那么成功配对，dp【i】=dp【i-1】+2
     * 但是还有可能，（）（（。。）这种情况，多了一个chars【i】直接让前面和dp【i-1】连接起来了，所以，
     * 还得加上前面的符合的长度所以dp【i】=dp【i-1】+2+dp【i-dp【i-1】-2】；
     * 最后处理一下边界条件就行了。
     * */
//    public int longestValidParentheses(String s) {
//        char[] chars = s.toCharArray();
//        if(s.length()==0)return 0;
//        int dp[] = new int[chars.length];
//        int max=0;
//        dp[0]=0;
//        for(int i=1;i<chars.length;i++){
//            if(chars[i]=='(') dp[i]=0;
//            else{
//                if(chars[i-1]=='('){
//                    if(i-2>-1) dp[i]=dp[i-2]+2;
//                    else dp[i]=2;
//                }
//                else{
//                    if(i-dp[i-1]-1>-1&&chars[i-dp[i-1]-1]=='('){
//                        if(i-dp[i-1]-2>-1){
//                        dp[i]=dp[i-dp[i-1]-2]+dp[i-1]+2;
//                        }
//                        else dp[i]= dp[i-1]+2;
//                    }
//                }
//            }
////            System.out.println(i+" :"+dp[i]);
//            if(dp[i]>max)max=dp[i];
//        }
//        return max;
//    }
    //31. 下一个排列
    /*
    数字的下一个排列方法，从后往前找，找到一个逆序对，说明，换了他俩的位置，这个数就能变大
    但是怎么确定他是临近的下一个呢？
    因为从后往前数都是顺序的了，所以再后边找到一个最小的数（进的是最小的），且满足逆序的数（数就变大），交换他俩位置，然后再给后面的数从小到大排好序，
    这时候最小（后面的是最小的排列）。
     */
//    pulic void nextPermutation(int[] nums) {
//        int temp;
//        for(int i=nums.length-1;i>0;i--){
//            if(nums[i]>nums[i-1]){
//                for(int j=nums.length-1;j>i-1;j--){
//                    if(nums[j]>nums[i-1]){
//                        temp=nums[j];
//                        nums[j]=nums[i-1];
//                        nums[i-1]=temp;
//
//                        Arrays.sort(nums, i, nums.length);
//                        return;
//                    }
//                }
//            }
//        }
//        Arrays.sort(nums);
//    }

    //20有效的括号
//    public boolean isValid(String s) {
//        //stack在Java中用双端队列的接口，实现类还是likedlist
//        Deque<Character> stack=new LinkedList<>();
//        char[] chars = s.toCharArray();
//        for(int i=0;i<chars.length;i++){
//            if (chars[i]=='{'||chars[i]=='('||chars[i]=='['){
//                stack.push(chars[i]);
//            }
//            else{
//                if(stack.isEmpty()){
//                    return false;
//                }
//                else{
//                    Character pop = stack.pop();
//                    if((pop=='{'&&chars[i]=='}')||(pop=='['&&chars[i]==']')||(pop=='('&&chars[i]==')')){
//                        continue;
//                    }
//                    else{
//                        return false;
//                    }
//                }
//            }
//        }
//        if(stack.isEmpty())
//            return true;
//        else return false;
//
//    }

    //19删除链表倒数第n个节点
//    public class ListNode {
//        int val;
//        ListNode next;
//        ListNode() {}
//        ListNode(int val) { this.val = val; }
//        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
//    }
//    public ListNode removeNthFromEnd(ListNode head, int n) {
//        ListNode last=head;
//        ListNode del=head;
//        ListNode predel=head;
//
//        for(int i=0;i<n;i++){
//            last=last.next;
//        }
//        //这个很重要，需要判定，是否删除的是头节点，是头节点直接返回head.next，因为，predel找不到。
//        int flag=0;
//        while (last!=null){
//            if(flag==0){
//                flag=1;
//
//
//                del=del.next;
//                last=last.next;
//            }
//            else{
//                predel=predel.next;
//                del=del.next;
//                last=last.next;
//            }
//        }
//        if(flag==0){
//            return head.next;
//        }
//        else {
//            predel.next = del.next;
//            return head;
//        }
//    }
    //    17.电话号码组合
    //回溯法，符合条件放入res中递归，深搜。
//    public List<String> letterCombinations(String digits) {
//
//        Map<Character,Character[]> map=new HashMap<>();
//        map.put('2',new Character[]{'a','b','c'} );
//        map.put('3',new Character[]{'d','e','f'} );
//        map.put('4',new Character[]{'g','h','i'} );
//        map.put('5',new Character[]{'j','k','l'} );
//        map.put('6',new Character[]{'m','n','o'} );
//        map.put('7',new Character[]{'p','q','r','s'} );
//        map.put('8',new Character[]{'t','u','v'} );
//        map.put('9',new Character[]{'w','x','y','z'} );
//        List<String> res=new ArrayList<>();
//        if(digits.length()==0){
//            return res;
//        }
//        backtrack(res, "", digits, 0, map);
//        return res;
//    }
//    public void backtrack(List<String> res,String combination,String digits,int index, Map<Character,Character[]> map){
//        if(digits.length()==index){
//            res.add(combination);
//        }
//        else{
//            for(int i=0;i<map.get(digits.charAt(index)).length;i++){
//                String s = combination+ map.get(digits.charAt(index))[i].toString();
//                backtrack(res, s, digits, index+1, map);
//            }
//
//        }
//    }
    //很简单的一个排列问题，用代码描述，就需要一些技巧了，可以通过队列来实现排列。
    //每个短的出队和需要新进来每个字母都进行结合，字母插在短的字符串后边。最后留在队列中的就是需要的串了
//    public List<String> letterCombinations(String digits) {
//        Map<Character,Character[]> map=new HashMap<>();
//        map.put('2',new Character[]{'a','b','c'} );
//        map.put('3',new Character[]{'d','e','f'} );
//        map.put('4',new Character[]{'g','h','i'} );
//        map.put('5',new Character[]{'j','k','l'} );
//        map.put('6',new Character[]{'m','n','o'} );
//        map.put('7',new Character[]{'p','q','r','s'} );
//        map.put('8',new Character[]{'t','u','v'} );
//        map.put('9',new Character[]{'w','x','y','z'} );
//        List<String> res=new ArrayList<>();
//        Queue<String> queue=new LinkedList<>();
//
//        for(int i = 0 ;i<digits.length();i++){
//            Character[] characters = map.get(digits.charAt(i));
//            if(i==0) {
//                for (Character character : characters) {
//                    queue.offer(String.valueOf(character));
//                }
//            }
//            else{
//                 while (queue.peek().length()==i){
//                     String poll=queue.poll();
//                     for(int j=0;j<characters.length;j++){
//                         String s=poll+characters[j].toString();
//                         queue.offer(s);
//                     }
//                 }
//            }
//
//        }
//        for (String s : queue) {
//            res.add(s);
//        }
//
//        return res;
//    }

 //             15. 三数之和
    /*
           用三个指针来代替三重循环，将O3的复杂度降到了O2
           1先排序
           选择一个指针不动，定点K ,再从后边的两边选i，j往中间遍历，遍历的过程中，排除许多不可能成立的，就降低了复杂度
           三重循环是枚举了所有可能，而双指针，则是把很多不可能的筛选出去了。
           例如1：如果nums[k]>0那么，就可以结束了，他后面怎么组合就不可能成功，并且，k是递增的，所以也不可能成功。
               2：如果nums[k-1]和nums[k]相等并且k-1比k还多一个选项k，所以，k有的，k-1都有，可以跳过这一次
               3：就是正常情况如何遍历了，如果sum小了，i就变大，如果sum大了j就变小往中间走。如果sum=0那么就存下来，
               并且让i，j都集中一次，因为已经用过了，需要的是不重复的组合。

     */
//    public List<List<Integer>> threeSum(int[] nums) {
//        List<List<Integer>> res = new ArrayList<List<Integer>>();
//        Arrays.sort(nums);
//        int k,i,j;
//        for(k=0;k<=nums.length-3;k++){
//            i=k+1;
//            j=nums.length-1;
//            //如果nums[k]==nums[k-1]说明，k-1的双指过的区域包括了k的，那么k有的k-1也有，所以就没必要再扫一遍了
//            if(k>0&&nums[k]==nums[k-1]) continue;
//            //如果nums[k]都比零大了，那么后面的nums[i],nums[j]就不可能为负，就不会组成三者合为0
//            if(nums[k]>0) break;
//
//            while (i<j){
//                int sum = nums[k]+nums[i]+nums[j];
//                if(sum<0) {
//                    int temp = nums[i];
//                    while (i <= nums.length - 1 && temp == nums[i]) {
//                        i++;
//                    }
//                }
//                else if(sum>0){
//                    int temp=nums[j];
//                    while (j>=k&&temp==nums[j]) {
//                        j--;
//                    }
//                }
//                else if(sum==0){
//                    ArrayList<Integer> integers = new ArrayList<>();
//                    integers.add(nums[k]);
//                    integers.add(nums[i]);
//                    integers.add(nums[j]);
//                    res.add(integers);
//                    int temp = nums[i];
//                    while (i <= nums.length - 1 && temp == nums[i]) {
//                        i++;
//                    }
//                    temp=nums[j];
//                    while (j>=k&&temp==nums[j]) {
//                        j--;
//                    }
//                }
//            }
//        }
//
//        return res;
//
//    }

         //11、盛最多的水的容器    (双指针其实也是暴力求解，不过是指针动态消去无效解来优化效率)
    //   x=Math.min(height[left],height[rigth])*Math.abs(rigth-left)
    //通过表达式可以看出，答案取决于两个变量，一个是最短边，一个是两个板的距离，距离可以控制，直接从两边开始找，这个时候
    //距离最长，那么下一步，只有两种办法，要么左边向里面移动，要么右边向里面移动，如果是长边移动，短边不变，那么height不变，或者
    //变小，距离也变小，结果肯定也是变小，所以，移动短边，可能下个边会长，导致结果增大，这样筛选去了很多没必要的枚举数据。
    //
//    public int maxArea(int[] height) {
//        int res=0;
//        int rigth=height.length-1;
//        int left=0;
//        while (left!=rigth){
//            int x=Math.min(height[left],height[rigth])*Math.abs(rigth-left);
//            if(x>res) res=x;
//            if(height[left]>height[rigth]){
//                rigth--;
//            }
//            else{
//                left++;
//            }
//        }
//        return res;
//    }
    //        10. 正则表达式匹配
//      用动态规划
//    public boolean isMatch(String s, String p) {
//        boolean [][]dp= new boolean[s.length()+1][p.length()+1];
//        //都是空串
//        dp[0][0]=true;
//        //初始化行
//         只有p为“a*a*”的时候才可以匹配空串
//        for(int j=2;j<=p.length();j+=2){
//            if(p.charAt(j-1)=='*'){
//                dp[0][j]=dp[0][j-2];
//            }
//        }
//        //列不需要初始化，当p为0，即为空串的时候，匹配不上任何s，除非也是空串。JavaBoolean数组默认为false；
//
//        for(int i=1;i<=s.length();i++){
//            for(int j=1;j<=p.length();j++){
//                //需要匹配的最后两个可以匹配，结果就取决于他们前面的能不能匹配
//                if (s.charAt(i-1)==p.charAt(j-1)||p.charAt(j-1)=='.'){
//                    dp[i][j]=dp[i-1][j-1];
//                }
//                //如果p是*号
//                else if(p.charAt(j-1)=='*'){
//                    //如果x！=si-1那么 x*就不能使用，且不为.x，舍弃这两个符号。
//                    if(p.charAt(j-2)!=s.charAt(i-1)&&p.charAt(j-2)!='.'){
//                        dp[i][j]=dp[i][j-2];
//                    }
//                    //可以使用，三种情况x*表示多个，一个，0个x只要有一个成功，那么就成功了
//                    else if (p.charAt(j-2)==s.charAt(i-1)||p.charAt(j-2)=='.'){
//                        dp[i][j]=dp[i-1][j]||dp[i][j-1]||dp[i][j-2];
//                    }
//                }
//
//            }
//        }
//        return dp[s.length()][p.length()] ;
//
//    }

    //                      5寻找最长回文子串（动态规划，枚举中心位置）
    //中心拓展法，动态规划其实也是暴力，还是在算是否匹配，枚举了每一个字符串，这个只枚举了所有的中心
    //虽然都是O(n2)但是这个更快一点。
/*    public String longestPalindrome(String s){

        int length = s.length();
        char[] chars = s.toCharArray();
        if(length<2){
            return s;
        }
        int longest=1,index=0;
        for(int i=0;i<=length-2;i++){
            int []i1 = lenByMid(chars, i, i);
            int []i2 = lenByMid(chars, i, i + 1);
            int len1=i1[1]-i1[0]+1;
            int len2=i2[1]-i2[0]+1;
            if(len1>len2){
                if(len1>longest){
                    index=i1[0];
                    longest=len1;
                }
            }
            else{
                if(len2>longest){
                    index=i2[0];
                    longest=len2;
                }
            }

        }
        return s.substring(index, index+longest);
    }
    public int[] lenByMid(char [] chars,int l,int r ){
        while(r>=0&&r<=chars.length-1 && l>=0&&l<=chars.length-1){
            if(chars[r]==chars[l]){
                l--;
                r++;
            }
            else{
                break;
            }
        }
        l++;
        r--;
        return new int[]{l,r};
    }*/
    //动态规划法
    public String longestPalindrome(String s) {

        char[] chars = s.toCharArray();
        int len=s.length();
        if(len<2)return s;

        boolean dp[][]= new boolean[len][len];
        for(int i=0;i<len;i++){
            for(int j=0;j<len;j++){
                dp[i][j]=false;
            }
        }
        for(int i =0;i<len;i++){
            dp[i][i]=true;
        }
        int longest=1,index=0;
        for(int j=1;j<len;j++){
            for(int i=0;i<j;i++){
                if(chars[i]==chars[j]){
                    if((j-1)-(i+1)+1<2){
                        dp[i][j]=true;
                    }
                    else {
                        dp[i][j]=dp[i+1][j-1];
                    }
                }
                else {
                    dp[i][j]=false;
                }
                if((j-i+1>longest)&&dp[i][j]){
                    longest= j-i+1;
                    index=i;
                }

            }
        }
        return s.substring(index, index+longest);

    }



    ////  							4寻找两个正序数组的中位数，
////也可以看作寻找第K小的数,通过算k/2的nums1和nums2，那个大，将小的那一部分的k/2舍去，因为一定不是第K个，可以舍去。
//当，k=1，或者一个数组为空时结束递归，所以，把len1，算为小的那个，起始点 i+1可能为end+1，所以end-（i+1）+1==0，这就得出数组为0了；
//class Solution {
//    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
//    	int left=(nums1.length+nums2.length+1)/2;
//    	int right=(nums1.length+nums2.length+2)/2;
//    	return (getKth(nums1,0,nums1.length-1,nums2,0,nums2.length-1,left)+getKth(nums1,0,nums1.length-1,nums2,0,nums2.length-1,right))*0.5;
//    }
//    int getKth(int [] nums1,int start1 ,int end1,int [] nums2,int start2,int end2,int k) {
//    	int len1=end1-start1+1;
//    	int len2=end2-start2+1;
//    	//每次让len1为小的，如果一个数组为空，一定是数组1；
//    	if(len1>len2) {
//    		return getKth(nums2,start2,end2,nums1,start1 , end1, k);
//    	}
//    	//一种，一个数组空了递归结束；
//    	if(len1==0) {
//    		return nums2[start2+k-1];
//    	}
//    	//k二分，二分，直至为1，直接就能选出结果了。结束。
//    	if(k==1) {
//    		return Math.min(nums1[start1], nums2[start2]);
//    	}
//    	int i=start1+Math.min(len1, k/2)-1;//这种方法，可以排除数组溢出，和记i下来，算出排除了多少个数字；
//    	int j=start2+Math.min(len2, k/2)-1;
//    	if(nums1[i]>=nums2[j]) {
//    		return getKth(nums1, start1, end1, nums2,j+1, end2, k-(j-start2+1) );
//    	}
//    	else {
//    		return getKth(nums1, i+1, end1, nums2,start2, end2, k-(i-start1+1) );
//    	}
//
//    }
//}
//                               632 最小区间
// 创建一个最小堆，每个数组选一个最小的元素进去，维护这个堆，则满足，这个堆的最大值和最小值是一个符合要求的区间；
//接下来的问题是找，最短区间。最大元素，只能增大，那么要想缩短，只能让最小元素增大，用max，标记每次入堆的元素的最大值；
//取出最小元素，看是否，max-min小于之前最优结果end-start；如果满足，就替换，不满足，就讲去除的最小元素原数组中的下一个元素
//放堆，因为，剔除了这个数组的元素，所以要再放一个这个数组的元素，最好的就是它下一个元素了，因为是有序的。记得比较是不是max。
//当剔除的那个元素的数组没有下一位时就可以结束了。因为这个堆无法维护下去了。最小元素，不能增加了，最大元素，只能增。
//class Solution {
//    public int[] smallestRange(List<List<Integer>> nums) {
//    	PriorityQueue<Node> pq=new PriorityQueue<Solution.Node>((o1,o2)->Integer.compare(o1.val,o2.val));
//    	int start,end;
//    	int max=-0x3f3f3f3f;
//    	for(int i=0;i<nums.size();i++) {
//    		Node node=new Node(i, 0, nums.get(i).get(0));
//    		max=Math.max(nums.get(i).get(0), max);
//    	}
//    	end=max;
//    	start=pq.peek().val;
//    	while(true) {
//    		Node node=pq.poll();
//    		if(max-node.val<end-start) {
//    			end=max;
//    			start=node.val;
//    		}
//    		if(node.i<nums.get(node.i).size()-1) {
//    			node=new Node(node.i,node.j+1,nums.get(node.i).get(node.j+1));
//    			max=Math.max(max,nums.get(node.i).get(node.j+1) );
//    			pq.offer(node);
//    		}
//    		else {
//    			break;
//    		}
//    	}
//    	return new int[]  {start,end};
//    }
//    class Node{
//    	int i;
//    	int j;
//    	int val;
//    	public Node(int i,int j,int val) {
//    		this.i=i;
//    		this.j=j;
//    		this.val=val;
//    	}
//    }
//}

//						567 字符串排列
// 简单的滑动窗口套路，通过两个指针维持一个表示字串的窗口，右边不断扩张添加新的元素，左边根据添加的元素，如果是不存在的，那么这个字串
//就不能用，直接放弃。左边调到最后面。如果存在，那么，判断这个字符是不是已经超过目标的个数了，如果超过了，那么，就收缩左边，直到不超过。

//class Solution {
//	public boolean checkInclusion(String s1, String s2) {
//		Map<Character, Integer> Maps1 = new HashMap<Character, Integer>();
//		Map<Character, Integer> Mapsub = new HashMap<Character, Integer>();
//		Boolean b = false;
//		int cnt = 0;
//		for (int i = 0; i < s1.length(); i++) {
//			Maps1.put(s1.charAt(i), Maps1.getOrDefault(s1.charAt(i), 0)+1);
//		}
//		int left = 0;
//		for (int i = 0; i < s2.length(); i++) {
//			Character c = s2.charAt(i);
//			if (!Maps1.containsKey(c)) {
//				Mapsub.clear();
//				left = i + 1;
//				cnt = 0;
//			}
//			else {
//				while (Mapsub.getOrDefault(c, 0) + 1 > Maps1.get(c)) {
//					Character x = s2.charAt(left);
//					left++;
//					Mapsub.put(x, Mapsub.getOrDefault(x, 0) - 1);
//					cnt--;
//				}
//				Mapsub.put(c, Mapsub.getOrDefault(c, 0) + 1);
//				cnt++;
//			}
//			if(cnt==s1.length()) {
//				b=true;
//				break;
//			}
//		}
//		return b;
//	}
//}

//										239滑动窗口最大值
//	窗口的复杂度低，是因为，一个列，进出个一次，所以，是线性的。
//	复杂度O（n）每个元素都进队出队各一次；
//	一个新元素，进队，判断，前面比他小的，肯定是没机会的元素，直接出队，这样，就不用重复比较大小了，这样，队列第一个就是最大的元素了
//但是队列第一个元素可能是不在窗口内的元素，所以，要存数组的下标，这样，可以判断，这个元素是不是要舍弃了；

//class Solution {
//	public int[] maxSlidingWindow(int[] nums, int k) {
//		Deque<Integer> deq = new ArrayDeque<Integer>();
//
//		int[] ans = new int[nums.length - k + 1];
//		int cnt = 0;
//
//		for (int i = 0; i < nums.length; i++) {
//			if (!deq.isEmpty() && deq.peekFirst() < (i - k + 1)) {
//				deq.pollFirst();
//			}
//
//			while (!deq.isEmpty() && nums[deq.peekLast()] <= nums[i]) {
//				deq.pollLast();
//			}
//			deq.offerLast(i);
//
//			if (i >= k - 1) {
//				ans[cnt++] = nums[deq.peekFirst()];
//			}
//		}
//		return ans;
//	}
//}

}

