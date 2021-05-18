import java.util.*;


class  Heap{
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
}



public class Main {
    public static void main(String[] args) {
        Main main = new Main();
        System.out.println(main.countSubstrings("abc"));

    }
    int[] preorder;
    int[] inorder;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length==0){
            return null;
        }
        this.preorder=preorder;
        this.inorder=inorder;
        return getTree(0,preorder.length-1,0,preorder.length-1);
    }
    TreeNode getTree(int i1,int j1,int i2,int j2){
        TreeNode treeNode = new TreeNode();
        int root=preorder[i1];
        treeNode.val=preorder[i1];
//        情况一，只有根节点，没有左右子树
        if(i1==j1){
            return treeNode;
        }
        else {
            int x=0;
            for(int i=i2;i<=j2;i++){
                if (root==inorder[i]){
                    x=i;
                    break;
                }
            }
//            情况二，没有左子树
            if(x==i2){
                treeNode.right=getTree(i1+1, j1, i2+1, j2);
            }
//            情况三，没有右子树
            else if(x==j2){
                treeNode.left=getTree(i1+1, j1, i2, j2-1);
            }
//            情况四，左右子树都有
            else {
                int left=x-i2;
                int right=j2-x;
                treeNode.left=getTree(i1+1, i1+1+left-1, i2, x-1);
                treeNode.right=getTree(j1-right+1, j1, x+1, j2);
            }

        }
        return treeNode;
    }
    public int[] reversePrint(ListNode head) {
        ListNode pre=null;
        while (head!=null){
            ListNode next= head.next;
            head.next=pre;
            pre=head;
            head=next;
        }
        ArrayList<Integer> list = new ArrayList<>();
        while (pre!=null){
            list.add(pre.val);
            pre=pre.next;
        }
        int []res=new int [list.size()];
        for(int i=0;i<res.length;i++){
            res[i]=list.get(i);
        }
        return res;
    }
    public String replaceSpace(String s) {
        StringBuffer sb=new StringBuffer(s);
        for(int i=0;i<sb.length();i++){
            if(sb.charAt(i)==' '){
                sb.deleteCharAt(i);
                sb.insert(i,"%20");
                i+=2;
            }
        }
        return sb.toString();
    }
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if(matrix.length<1){
            return false;
        }
        int n=matrix.length;
        int m=matrix[0].length;
        int i=0,j=m-1;
        while (i>=0&&j>=0&&i<n&&j<m){
            if(matrix[i][j]==target){
                return true;
            }
            else {
                if(target>matrix[i][j]){
                    i++;
                }
                else {
                    j--;
                }
            }
        }
        return false;
    }
    public int findRepeatNumber(int[] nums) {
        int n=nums.length;
        for(int i=0;i<nums.length;i++){
            int x=nums[i]%n;
//            标记x位置，x已经存在
            nums[x]+=n;
        }
        int res=0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]>=2*n){
                res=i;
                break;
            }
        }
        return res;
    }
    public int[] dailyTemperatures(int[] temperatures) {
        int [] res = new int[temperatures.length];
        Deque<Integer> stack = new LinkedList<>();
        for(int i=0;i<temperatures.length;i++){
            while (!stack.isEmpty()){
                Integer peek = stack.peek();
                if(temperatures[peek]<temperatures[i]){
                    res[peek]=i-peek;
                    stack.pop();
                }
                else {
                    break;
                }
            }
            stack.push(i);
        }
        return res;
    }
    public int countSubstrings(String s) {
        int length = s.length();
        char[] chars = s.toCharArray();
        if(length<2){
            return 1;
        }
        int res=0;
        for(int i=0;i<=length-2;i++){
            int []i1 = lenByMid(chars, i, i);
            int []i2 = lenByMid(chars, i, i + 1);
            int len1=i1[1]-i1[0]+1;
            int len2=i2[1]-i2[0]+1;
            if(len1!=0){
                if(len1==1){
                    res+=len1;
                }
                else {
                    if(len1%2==0){
                        res+=len1/2;
                    }
                    else {
                        res+=len1/2+1;
                    }
                }
            }
            if(len2!=0){
                if(len2==1){
                    res+=len2;
                }
                else {
                    if(len2%2==0){
                        res+=len2/2;
                    }
                    else {
                        res+=len2/2+1;
                    }
                }
            }

        }
        return res+1;
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
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1==null){
            return root2;
        }
        if(root2==null){
            return root1;
        }
        TreeNode node = new TreeNode(root1.val+root2.val);
        node.left=mergeTrees(root1.left, root2.left);
        node.right=mergeTrees(root1.right, root2.right);
        return node;
    }


//    560. 和为K的子数组
//    前缀和，哈希表
    public int subarraySum(int[] nums, int k) {

        int count=0;
        Map<Integer,Integer> map = new HashMap<>();
//        前缀为空，则前缀和为0，数目是1；
        map.put(0, 1);
        int prefixSum=0;
        for(int i=0;i<nums.length;i++){
            prefixSum+=nums[i];
//            当前这个数，和前面某个数的前缀和相减等于k
            if(map.containsKey(prefixSum-k)){
                count+=map.get(prefixSum-k);
            }
            map.put(prefixSum,map.getOrDefault(prefixSum, 0)+1);
        }
        return count;
    }
    public int hammingDistance(int x, int y) {
        int w = x^y;
        int count=0;
        while (w>=1){
            if(w%2==1){
                count++;
            }
            w>>=1;
        }
        return count;
    }
    Map<TreeNode,Integer> map=new HashMap<>();
    public int diameterOfBinaryTree(TreeNode root) {
        int max=0;
        Queue<TreeNode> queue=new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            TreeNode poll = queue.poll();
            max=Math.max(max, height(poll.left)+height(poll.right));
            if(poll.left!=null) queue.add(poll.left);
            if(poll.right!=null) queue.add(poll.right);
        }
        return max;
    }
    private int height(TreeNode node){
        if(node==null)return 0;

        else {
            if(map.containsKey(node)){
                return map.get(node);
            }
            else {
                map.put(node,Math.max(height(node.left), height(node.right))+1);
                return map.get(node);
            }
        }
    }
    public List<Integer> findAnagrams(String s, String p) {
        if(p.length()>s.length()){
            return new ArrayList<>();
        }
        Map<Character,Integer> mapp=new HashMap<>();
        for (Character c:p.toCharArray()){
            mapp.put(c, mapp.getOrDefault(c, 0)+1);
        }

        Map<Character,Integer> maps=new HashMap<>();
        for(int i=0;i<p.length();i++ ){
            maps.put(s.charAt(i),maps.getOrDefault(s.charAt(i), 0)+1);
        }
        int i=0,j=p.length()-1;
        List<Integer> ans = new ArrayList<>();
        if(maps.equals(mapp)){
            ans.add(i);
        }
        i++;
        j++;
        while (j<s.length()){
            char c = s.charAt(i - 1);
            Integer integer = maps.get(c);
            if(integer!=1){
                maps.put(c, integer-1);
            }
            else {
                maps.remove(c);
            }
            maps.put(s.charAt(j), maps.getOrDefault(s.charAt(j), 0)+1);
            if(mapp.equals(maps)){
                ans.add(i);
            }
            i++;
            j++;
        }
        return ans;
    }




//    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
//        int equationsSize = equations.size();
//
//        UnionFind unionFind = new UnionFind(2 * equationsSize);
//        // 第 1 步：预处理，将变量的值与 id 进行映射，使得并查集的底层使用数组实现，方便编码
//        Map<String, Integer> hashMap = new HashMap<>(2 * equationsSize);
//        int id = 0;
//        for (int i = 0; i < equationsSize; i++) {
//            List<String> equation = equations.get(i);
//            String var1 = equation.get(0);
//            String var2 = equation.get(1);
//
//            if (!hashMap.containsKey(var1)) {
//                hashMap.put(var1, id);
//                id++;
//            }
//            if (!hashMap.containsKey(var2)) {
//                hashMap.put(var2, id);
//                id++;
//            }
//            unionFind.union(hashMap.get(var1), hashMap.get(var2), values[i]);
//        }
//
//        // 第 2 步：做查询
//        int queriesSize = queries.size();
//        double[] res = new double[queriesSize];
//        for (int i = 0; i < queriesSize; i++) {
//            String var1 = queries.get(i).get(0);
//            String var2 = queries.get(i).get(1);
//
//            Integer id1 = hashMap.get(var1);
//            Integer id2 = hashMap.get(var2);
//
//            if (id1 == null || id2 == null) {
//                res[i] = -1.0d;
//            } else {
//                res[i] = unionFind.isConnected(id1, id2);
//            }
//        }
//        return res;
//    }

//    private class UnionFind {
//
//        private int[] parent;
//
//        /**
//         * 指向的父结点的权值
//         */
//        private double[] weight;
//
//
//        public UnionFind(int n) {
//            this.parent = new int[n];
//            this.weight = new double[n];
//            //初始化
//            for (int i = 0; i < n; i++) {
//                parent[i] = i;
//                weight[i] = 1.0d;
//            }
//        }
//
//        public void union(int x, int y, double value) {
//            int rootX = find(x);
//            int rootY = find(y);
//            if (rootX == rootY) {
//                return;
//            }
//
//            parent[rootX] = rootY;
//            // 关系式的推导请见「参考代码」下方的示意图
//            weight[rootX] = weight[y] * value / weight[x];
//        }
//
//        /**
//         * 路径压缩
//         *
//         * @param x
//         * @return 根结点的 id
//         */
//        public int find(int x) {
//            if (x != parent[x]) {
//                int origin = parent[x];
//                parent[x] = find(parent[x]);
//                weight[x] *= weight[origin];
//            }
//            return parent[x];
//        }
//
//        public double isConnected(int x, int y) {
//            int rootX = find(x);
//            int rootY = find(y);
//            if (rootX == rootY) {
//                return weight[x] / weight[y];
//            } else {
//                return -1.0d;
//            }
//        }
//    }











    public int findDuplicate(int[] nums) {
        int slow = 0 ;
        int fast= 0 ;
        //都往后走了一步
        slow=nums[slow];
        fast=nums[nums[fast]];
        while (slow!=fast){
            slow=nums[slow];
            fast=nums[nums[fast]];
        }
        slow=0;
        while (slow!=fast){
            slow=nums[slow];
            fast=nums[fast];
        }

        return slow;
    }
    public void moveZeroes(int[] nums) {
        int left=-1;
        int right=0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]==0){
                left=i;
                break;
            }
        }
        if(left==-1)return;
        right=left+1;
        while (right!=nums.length){
            if(nums[right]==0){
                right++;
            }
            else {
                swap(right, left, nums);
                left++;
                right++;
            }
        }
    }
    void swap(int i,int j,int[] nums){
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }


    public boolean searchMatrix(int[][] matrix, int target) {
        int m=matrix.length;
        int n=matrix[0].length;
        return find(m-1, 0, matrix, target);
    }
    boolean find(int i,int j,int [][] matrix,int target){
        if(matrix[i][j]==target){
            return true;
        }
        else if(matrix[i][j]<target){
            j++;
            if(j==matrix[0].length){
                return false;
            }
            else {
                return find(i,j,matrix,target);
            }
        }
        else{
            i--;
            if(i==-1){
                return false;
            }
            else {
                return find(i,j,matrix,target);
            }
        }
    }
    public int[] productExceptSelf(int[] nums) {

        int []ans=new int [nums.length];
        int prefix=1;
        for(int i=0;i<nums.length;i++){
            ans[i]=prefix;
            prefix=prefix*nums[i];
        }
        int suffix=1;
        for(int i=nums.length-1;i>=0;i--){
            ans[i]*=suffix;
            suffix*=nums[i];
        }
        return ans;
    }


    public boolean isPalindrome(ListNode head) {
        if(head.next==null){
            return true;
        }
        int count=0;
        ListNode cur=head;
        while (cur!=null){
            count++;
            cur=cur.next;
        }
        ListNode fast=head.next;
        ListNode slow=head;
        while (fast!=null){
            slow=slow.next;
            fast=fast.next;
            if(fast!=null){
                fast=fast.next;
            }
        }
        ListNode head2;
        if(count%2==0){
            head2=slow;
        }
        else {
            head2=slow.next;
        }

        head2=reverse(head2);
        while (head2!=null){
            if(head2.val==head.val){
                head=head.next;
                head2=head2.next;
            }
            else {
                return false;
            }
        }
        return true;
    }
    ListNode reverse(ListNode head){
        ListNode pre=null;
        while (head!=null){
            ListNode temp=head.next;
            head.next=pre;
            pre=head;
            head=temp;
        }
        return pre;
    }








    public TreeNode invertTree(TreeNode root) {
        if(root==null){
            return null;
        }
        TreeNode tem=root.left;
        root.left=root.right;
        root.right=tem;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }







    public int findKthLargest(int[] nums, int k) {
        Heap heap = new Heap(nums.length);
        heap.createHeap(nums);
        for(int i=0;i<k;i++){
            int pop = heap.pop();
            if(i==k-1)return pop;
        }
        return 0;
    }

    public int numIslands(char[][] grid) {
        int m=grid.length,n=grid[0].length;
        int G[][]=new int[m][n];
        int count=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if (dfs(grid, G, i, j)){
                     count++;
                }

            }
        }
        return count;
    }
    boolean dfs(char [][] grid,int G[][],int i,int j){
        if(G[i][j]==1||grid[i][j]=='0'){
            return false;
        }
        else {
            G[i][j]=1;
            if(i+1<grid.length){
                dfs(grid, G, i+1, j);
            }
            if(i-1>=0){
                dfs(grid, G, i-1, j);
            }
            if(j-1>=0){
                dfs(grid, G, i, j-1);
            }
            if(j+1<grid[0].length){
                dfs(grid, G, i, j+1);
            }
            return true;
        }
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
    public int maxProduct(int[] nums) {
        int max=Integer.MIN_VALUE;
        int iMin=1,iMax=1;
        for (int num : nums) {
            //为负数，就交换最大值和最小值，这样，最大值是负数，与负数相乘，才可能得到最大值
            if(num<0){
                int temp=iMax;
                iMax=iMin;
                iMin=temp;
            }
            //如果*num没有num本身大，那么就从num开始计算。
            iMax=Math.max(iMax*num, num);
            iMin=Math.min(iMin*num, num);
            max= Math.max(max, iMax);
        }
        return max;
    }

    class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

}

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
/*
* 用两个栈，一个保存这个位置的value，一个保持这个位置对应的最小的数，加入的时候都加入，弹出的时候都弹出
* */
class MinStack {

    private Deque<Integer> stack=new LinkedList<>();
    private Deque<Integer> min=new LinkedList<>();
    /** initialize your data structure here. */
    public MinStack() {

    }

    public void push(int val) {
        stack.push(val);
        if(min.isEmpty()){
            min.push(val);
        }
        else {
            min.push(Math.min(min.peek(),val));
        }
    }

    public void pop() {
        min.pop();
        stack.pop();

    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return min.peek();
    }
}
