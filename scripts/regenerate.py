#!/usr/bin/env python3
"""
Regenerate low-quality entries in final_generated.jsonl.
Replaces placeholder think/answer with proper Arabic content and Python code.
"""

import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

RAW = DATA_DIR / "raw.jsonl"
FINAL = DATA_DIR / "final_generated.jsonl"


def clean(line):
    line = line.strip()
    if line.startswith("//"):
        line = line[2:].strip()
    return line


def load_raw():
    entries = {}
    with open(RAW, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            c = clean(line)
            if c:
                try:
                    entries[i] = json.loads(c)
                except:
                    pass
    return entries


def load_final():
    entries = {}
    with open(FINAL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    entries[int(obj["id"])] = obj
                except:
                    pass
    return entries


INSTRUCT_TO_TOPIC = {
    "permute": "استدعاء ذاتي",
    "factorial": "استدعاء ذاتي",
    "fibonacci": "استدعاء ذاتي",
    "recursive": "استدعاء ذاتي",
    "sum_digits": "استدعاء ذاتي",
    "max_depth": "استدعاء ذاتي",
    "backtrack": "Backtracking",
    "combination": "Backtracking",
    "subset": "Backtracking",
    "n_queen": "Backtracking",
    "bfs": "طوابير",
    "shortest_path": "رسوم بيانية",
    "cycle": "رسوم بيانية",
    "graph": "رسوم بيانية",
    "dfs": "رسوم بيانية",
    "binary_search": "بحث",
    "search": "بحث",
    "find_element": "بحث",
    "find_missing": "بحث",
    "first_occurrence": "بحث",
    "last_occurrence": "بحث",
    "sliding": "نافذة منزلقة",
    "max_subarray": "نافذة منزلقة",
    "window": "نافذة منزلقة",
    "unique_chars": "نافذة منزلقة",
    "longest_substring": "نافذة منزلقة",
    "dp": "برمجة ديناميكية",
    "dynamic": "برمجة ديناميكية",
    "count_paths": "برمجة ديناميكية",
    "grid": "برمجة ديناميكية",
    "coin_change": "برمجة ديناميكية",
    "lis": "برمجة ديناميكية",
    "knapsack": "برمجة ديناميكية",
    "gcd": "رياضيات",
    "lcm": "رياضيات",
    "prime": "رياضيات",
    "divisors": "رياضيات",
    "binary": "رياضيات",
    "decimal_to_binary": "رياضيات",
    "power": "رياضيات",
    "is_palindrome": "سلاسل نصية",
    "palindrome": "سلاسل نصية",
    "anagram": "سلاسل نصية",
    "compress": "سلاسل نصية",
    "snake_to_camel": "سلاسل نصية",
    "longest_word": "سلاسل نصية",
    "most_frequent_char": "سلاسل نصية",
    "balanced": "مكدسات",
    "stack": "مكدسات",
    "postfix": "مكدسات",
    "queue": "طوابير",
    "two_sum": "مؤشرين",
    "pair_sum": "مؤشرين",
    "remove_duplicates": "مؤشرين",
    "merge_sorted": "مؤشرين",
    "activity": "جشع",
    "interval": "جشع",
    "greedy": "جشع",
    "sort": "ترتيب",
    "intersection": "مجموعات",
    "dict": "قواميس",
    "dictionary": "قواميس",
    "merge_dict": "قواميس",
}

# Answer generators by function name
ANSWERS = {}


def answer_for(func_name):
    def dec(fn):
        ANSWERS[func_name] = fn
        return fn
    return dec


@answer_for("split_list")
def _(instr):
    return """def split_list(head):
    if not head or not head.next:
        return head
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    return second"""


@answer_for("generate_permutations")
def _(instr):
    return """def generate_permutations(nums):
    res = []
    nums = list(nums)
    def bt(s):
        if s == len(nums):
            res.append(list(nums))
            return
        for i in range(s, len(nums)):
            nums[s], nums[i] = nums[i], nums[s]
            bt(s + 1)
            nums[s], nums[i] = nums[i], nums[s]
    bt(0)
    return res"""


@answer_for("max_unique_chars")
def _(instr):
    return """def max_unique_chars(s, k):
    from collections import Counter
    if len(s) < k: return 0
    cnt = Counter(s[:k])
    mx = len(cnt)
    for i in range(k, len(s)):
        cnt[s[i]] += 1
        cnt[s[i-k]] -= 1
        if cnt[s[i-k]] == 0:
            del cnt[s[i-k]]
        mx = max(mx, len(cnt))
    return mx"""


@answer_for("count_paths")
def _(instr):
    return """def count_paths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]"""


@answer_for("factorial")
def _(instr):
    return """def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)"""


@answer_for("find_pair_sum")
def _(instr):
    return """def find_pair_sum(nums, target):
    seen = set()
    for x in nums:
        c = target - x
        if c in seen:
            return sorted([x, c])
        seen.add(x)
    return []"""


@answer_for("partial_sort")
def _(instr):
    return """def partial_sort(arr, start, end):
    arr = list(arr)
    arr[start:end+1] = sorted(arr[start:end+1])
    return arr"""


@answer_for("min_operations")
def _(instr):
    return """def min_operations(target, current):
    return abs(target - current)"""


@answer_for("shortest_path")
def _(instr):
    return """from collections import deque

def shortest_path(graph, start, end):
    if start == end: return 0
    v = {start}
    q = deque([(start, 0)])
    while q:
        n, d = q.popleft()
        for nb in graph.get(n, []):
            if nb == end: return d + 1
            if nb not in v:
                v.add(nb)
                q.append((nb, d + 1))
    return -1"""


@answer_for("sliding_average")
def _(instr):
    return """def sliding_average(nums, k):
    if not nums or k > len(nums): return []
    s = sum(nums[:k])
    r = [s / k]
    for i in range(k, len(nums)):
        s += nums[i] - nums[i - k]
        r.append(s / k)
    return r"""


@answer_for("gcd")
def _(instr):
    return """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a"""


@answer_for("is_balanced")
def _(instr):
    return """def is_balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack.pop() != pairs[ch]:
                return False
    return len(stack) == 0"""


@answer_for("find_element")
def _(instr):
    return """def find_element(arr, target):
    return target in arr"""


@answer_for("coin_change")
def _(instr):
    return """def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1"""


@answer_for("compress_string")
def _(instr):
    return """def compress_string(s):
    if not s: return ''
    res, cnt = [], 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            cnt += 1
        else:
            res.append(s[i-1] + str(cnt))
            cnt = 1
    res.append(s[-1] + str(cnt))
    return ''.join(res)"""


@answer_for("get_divisors")
def _(instr):
    return """def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]"""


@answer_for("common_values")
def _(instr):
    return """def common_values(d1, d2):
    return list(set(d1.values()) & set(d2.values()))"""


@answer_for("pattern_match")
def _(instr):
    return """def pattern_match(text, pattern):
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if pattern[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif pattern[j-1] == '?' or pattern[j-1] == text[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]"""


@answer_for("evaluate_postfix")
def _(instr):
    return """def evaluate_postfix(tokens):
    st = []
    for t in tokens:
        if t in '+-*/':
            b, a = st.pop(), st.pop()
            if t == '+': st.append(a + b)
            elif t == '-': st.append(a - b)
            elif t == '*': st.append(a * b)
            else: st.append(int(a / b))
        else:
            st.append(int(t))
    return st[0]"""


@answer_for("is_full")
def _(instr):
    return """def is_full(queue, size):
    return len(queue) == size"""


@answer_for("fibonacci")
def _(instr):
    return """def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""


@answer_for("longest_consecutive_sublist")
def _(instr):
    return """def longest_consecutive_sublist(lst):
    if not lst: return []
    mx, cur, start = 1, 1, 0
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1]:
            cur += 1
        else:
            if cur > mx:
                mx = cur
                start = i - mx
            cur = 1
    if cur > mx:
        mx = cur
        start = len(lst) - mx
    return lst[start:start+mx]"""


@answer_for("lcm")
def _(instr):
    return """def lcm(a, b):
    def gcd(x, y):
        while y: x, y = y, x % y
        return x
    return a * b // gcd(a, b)"""


@answer_for("bfs")
def _(instr):
    return """from collections import deque

def bfs(graph, start):
    v, q, r = {start}, deque([start]), []
    while q:
        n = q.popleft()
        r.append(n)
        for nb in graph.get(n, []):
            if nb not in v:
                v.add(nb)
                q.append(nb)
    return r"""


@answer_for("merge_dicts")
def _(instr):
    return """def merge_dicts(d1, d2):
    r = dict(d1)
    for k, v in d2.items():
        r[k] = r.get(k, 0) + v
    return r"""


@answer_for("find_missing_number")
def _(instr):
    return """def find_missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)"""


@answer_for("sum_digits_recursive")
def _(instr):
    return """def sum_digits_recursive(n):
    if n == 0: return 0
    return n % 10 + sum_digits_recursive(n // 10)"""


@answer_for("fibonacci_dp")
def _(instr):
    return """def fibonacci_dp(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]"""


@answer_for("binary_search")
def _(instr):
    return """def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == target: return m
        elif arr[m] < target: l = m + 1
        else: r = m - 1
    return -1"""


@answer_for("decimal_to_binary")
def _(instr):
    return """def decimal_to_binary(n):
    if n == 0: return '0'
    res = ''
    while n > 0:
        res = str(n % 2) + res
        n //= 2
    return res"""


@answer_for("rotate_list")
def _(instr):
    return """def rotate_list(nums, k):
    if not nums: return []
    k %= len(nums)
    return nums[-k:] + nums[:-k]"""


@answer_for("most_frequent_char")
def _(instr):
    return """def most_frequent_char(s):
    from collections import Counter
    c = Counter(s)
    mx = max(c.values())
    return min(k for k, v in c.items() if v == mx)"""


@answer_for("max_intervals_scheduled")
def _(instr):
    return """def max_intervals_scheduled(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[1])
    cnt, end = 1, intervals[0][1]
    for s, e in intervals[1:]:
        if s >= end:
            cnt += 1
            end = e
    return cnt"""


@answer_for("is_palindrome")
def _(instr):
    return """def is_palindrome(s):
    return s == s[::-1]"""


@answer_for("first_occurrence")
def _(instr):
    return """def first_occurrence(s, ch):
    for i, c in enumerate(s):
        if c == ch: return i
    return -1"""


@answer_for("count_connected_components")
def _(instr):
    return """def count_connected_components(adj, n):
    v = set()
    def dfs(node):
        v.add(node)
        for nb in adj[node]:
            if nb not in v: dfs(nb)
    cnt = 0
    for i in range(n):
        if i not in v:
            dfs(i)
            cnt += 1
    return cnt"""


@answer_for("reverse_stack")
def _(instr):
    return """def reverse_stack(stack):
    if not stack: return
    top = stack.pop()
    reverse_stack(stack)
    stack.insert(0, top)
    return stack"""


@answer_for("extract_patterns")
def _(instr):
    return """def extract_patterns(s):
    res = []
    for i in range(len(s) - 1):
        if s[i] == s[i+1]:
            res.append(s[i] + s[i+1])
    return res"""


@answer_for("count_letters")
def _(instr):
    return """def count_letters(d):
    return {k: len(v) for k, v in d.items()}"""


@answer_for("fibonacci_sum")
def _(instr):
    return """def fibonacci_sum(n):
    if n <= 1: return 0
    a, b, s = 0, 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
        s += b
    return s if n >= 2 else 0"""


@answer_for("is_prime")
def _(instr):
    return """def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    i = 3
    while i * i <= n:
        if n % i == 0: return False
        i += 2
    return True"""


@answer_for("next_greater_element")
def _(instr):
    return """def next_greater_element(nums):
    res = [-1] * len(nums)
    st = []
    for i in range(len(nums)):
        while st and nums[i] > nums[st[-1]]:
            res[st.pop()] = nums[i]
        st.append(i)
    return res"""


@answer_for("find_substring_indices")
def _(instr):
    return """def find_substring_indices(text, pattern):
    res, i = [], 0
    while True:
        i = text.find(pattern, i)
        if i == -1: break
        res.append(i)
        i += 1
    return res"""


@answer_for("sort_by_length")
def _(instr):
    return """def sort_by_length(strings):
    return sorted(strings, key=len)"""


@answer_for("find_unique_elements")
def _(instr):
    return """def find_unique_elements(lst):
    from collections import Counter
    return [k for k, v in Counter(lst).items() if v == 1]"""


@answer_for("merge_two_lists")
def _(instr):
    return """def merge_two_lists(l1, l2):
    res = []
    i = j = 0
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            res.append(l1[i]); i += 1
        else:
            res.append(l2[j]); j += 1
    res.extend(l1[i:])
    res.extend(l2[j:])
    return res"""


@answer_for("sort_students")
def _(instr):
    return """def sort_students(students):
    return sorted(students, key=lambda x: (-x[1], x[0]))"""


@answer_for("solve_n_queens")
def _(instr):
    return """def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == row - i:
                return False
        return True
    def bt(row):
        if row == n:
            res[0] += 1
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                bt(row + 1)
    board, res = [-1] * n, [0]
    bt(0)
    return res[0]"""


@answer_for("custom_sort")
def _(instr):
    return """def custom_sort(arr):
    ev = sorted(x for x in arr if x % 2 == 0)
    od = sorted((x for x in arr if x % 2 != 0), reverse=True)
    return ev + od"""


@answer_for("activity_selection")
def _(instr):
    return """def activity_selection(start, finish):
    n = len(start)
    acts = sorted(zip(start, finish), key=lambda x: x[1])
    cnt, end = 1, acts[0][1]
    for s, e in acts[1:]:
        if s >= end:
            cnt += 1
            end = e
    return cnt"""


@answer_for("subsets")
def _(instr):
    return """def subsets(nums):
    res = [[]]
    for n in nums:
        res += [s + [n] for s in res]
    return res"""


@answer_for("subsets_backtracking")
def _(instr):
    return """def subsets_backtracking(nums):
    res = []
    def bt(s, cur):
        res.append(list(cur))
        for i in range(s, len(nums)):
            cur.append(nums[i])
            bt(i + 1, cur)
            cur.pop()
    bt(0, [])
    return res"""


@answer_for("last_occurrence")
def _(instr):
    return """def last_occurrence(lst, target):
    for i in range(len(lst)-1, -1, -1):
        if lst[i] == target: return i
    return -1"""


@answer_for("prime_factors")
def _(instr):
    return """def prime_factors(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1: f.append(n)
    return f"""


@answer_for("multiple")
def _(instr):
    return """def multiples(n, limit):
    return list(range(n, limit + 1, n))"""


@answer_for("intersection")
def _(instr):
    return """def intersection(list1, list2):
    s2 = set(list2)
    res, seen = [], set()
    for x in list1:
        if x in s2 and x not in seen:
            res.append(x)
            seen.add(x)
    return res"""


# ---- THINK TEMPLATES ----
THINKS = {}


def think_for(name):
    def dec(fn):
        THINKS[name] = fn
        return fn
    return dec


@think_for("balanced")
def _(instr):
    return "المطلوب التحقق من توازن الأقواس في سلسلة نصية. الفكرة الأساسية هي استخدام مكدس: كلما صادفنا قوسا مفتوحا نضيفه إلى المكدس، وكلما صادفنا قوسا مغلقا نتحقق من أن رأس المكدس هو القوس المقابل له. إذا كان التطابق صحيحا نزيل القوس المفتوح من المكدس، وإلا فالسلسلة غير متوازنة. في النهاية، إذا كان المكدس فارغا فجميع الأقواس متوازنة. نستخدم قاموسا يربط كل قوس مغلق بالقوس المفتوح المناظر له لتسهيل المقارنة. الحالة الفارغة تعيد True لأن السلسلة الفارغة تعتبر متوازنة تلقائيا."


@think_for("permutations")
def _(instr):
    return "المطلوب توليد جميع التباديل الممكنة لقائمة من الأرقام. عدد التباديل لقائمة طولها n هو n! وكل تبديل هو ترتيب مختلف لجميع العناصر. الفكرة الأساسية هي الاستدعاء الذاتي مع التبديل في المكان: في كل مستوى نثبت عنصراً في الموضع الحالي ونتبادل مع كل عنصر في المواضع اللاحقة، ثم نستدعي الدالة على الموضع التالي، ثم نعيد التبديل لاستعادة الحالة الأصلية. الحالة الأساسية هي حين يصل المؤشر إلى طول القائمة فنحفظ نسخة من التبديل الحالي. حالة القائمة الفارغة تعيد [[]]، وحالة العنصر الواحد تعيد [[1]]."


@think_for("factor")
def _(instr):
    return "المطلوب حساب n! باستخدام الاستدعاء الذاتي. العلاقة التعاودية واضحة: n! = n في n-1! لأن ضرب الأعداد من 1 إلى n يساوي n مضروبا في ضرب الأعداد من 1 إلى n-1. الحالة الأساسية هي n أقل من أو يساوي 1 حيث نعيد 1 لأن 0! و 1! يساويان 1. هذا المثال الكلاسيكي يوضح كيفية تقسيم المشكلة إلى أجزاء أصغر من نفس النوع حتى نصل إلى حالة يمكن حلها مباشرة."


@think_for("gcd")
def _(instr):
    return "المطلوب حساب القاسم المشترك الأكبر باستخدام خوارزمية إقليدس. تعتمد الخوارزمية على المبدأ التالي: القاسم المشترك الأكبر للعددين a و b يساوي القاسم المشترك الأكبر لـ b وباقي قسمة a على b. نكرر هذه العملية حتى يصبح b صفرا، عندها يكون a هو القاسم المشترك الأكبر. مثلا GCD 48 و 18: 48 mod 18 = 12، GCD 18 و 12: 18 mod 12 = 6، GCD 12 و 6: 12 mod 6 = 0، نعيد 6. الخوارزمية فعالة جدا بتعقب O(log min(a,b))."


@think_for("coin")
def _(instr):
    return "المطلوب إيجاد أقل عدد من العملات لتكوين مبلغ معين. الفكرة الأساسية هي البرمجة الديناميكية الصاعدة: نعرف dp[i] بأنه أقل عدد من العملات لتكوين المبلغ i. نبدأ من dp[0] = 0، ثم لكل مبلغ i من 1 حتى amount، نجرب كل عملة لا تتجاوز قيمتها i، ونأخذ الحد الأدنى بين dp[i-coin] + 1. هذا يعمل لأن مسألة الصرف تحقق خاصية الحل الأمثل للمسائل الفرعية: الحل الأمثل لمبلغ i يمكن بناؤه من الحل الأمثل لمبلغ أصغر مضافا إليه عملة واحدة."


@think_for("bfs")
def _(instr):
    return "المطلوب تنفيذ البحث بالعرض BFS على رسم بياني. نبدأ من عقدة البداية ونستخدم طابورا لاستكشاف العقد طبقة بعد طبقة. لكل عقدة نزورها نضيف جيرانها غير المزارين إلى الطابور. نستخدم مجموعة visited لتجنب معالجة العقد مرتين. BFS يضمن الوصول إلى كل عقدة بأقل عدد من الخطوات في الرسوم البيانية غير الموزونة. الترتيب النهائي للعقد هو ترتيب زيارتها حسب بعدها عن نقطة البداية."


@think_for("dfs")
def _(instr):
    return "المطلوب تنفيذ البحث بالعمق DFS على رسم بياني. نستخدم الاستدعاء الذاتي لاستكشاف العقد: نبدأ من عقدة، نضع علامة عليها كمزارة، ثم نستدعي الدالة على كل جار لم يزر بعد. نستخدم مجموعة visited لتجنب إعادة زيارة العقد. DFS تستكشف المسار حتى النهاية قبل التراجع، مما يجعلها مناسبة لمسائل مثل الكشف عن الدورات والترتيب الطوبولوجي."


@think_for("subset")
def _(instr):
    return "المطلوب توليد جميع المجموعات الجزئية الممكنة من قائمة. الفكرة التكرارية: نبدأ بنتيجة تحتوي على المجموعة الفارغة فقط. لكل عنصر جديد في القائمة، نأخذ جميع المجموعات الموجودة ونضيف إليها نسخا جديدة تحتوي على هذا العنصر. بعد معالجة جميع العناصر نحصل على 2^n مجموعة جزئية. هذا الأسلوب واضح وبسيط ويعمل بزمن O(n*2^n)."


@think_for("binary_search")
def _(instr):
    return "المطلوب تنفيذ البحث الثنائي في مصفوفة مرتبة. الفكرة هي تقليص مساحة البحث إلى النصف في كل خطوة: نحدد منتصف المصفوفة، إذا تطابق مع الهدف نعيد فهرسه، إذا كان الهدف أكبر نتجاهل النصف الأيسر ونبحث في النصف الأيمن، والعكس صحيح. نكرر حتى نجد الهدف أو يتقاطع المؤشران. التعقيد الزمني O(log n) وهو أفضل بكثير من البحث الخطي O(n) للمصفوفات الكبيرة."


@think_for("compression")
def _(instr):
    return "المطلوب ضغط سلسلة نصية بتحويل كل مجموعة من الحروف المتتالية المتشابهة إلى الحرف تبعه عدد تكراراته. الفكرة: نمر على السلسلة من اليسار إلى اليمين ونتتبع عدد تكرار الحرف الحالي. عندما يتغير الحرف نضيف الحرف السابق مع عدده إلى النتيجة. ننتهي بإضافة آخر مجموعة بعد انتهاء الحلقة. مثلا aabcccccaaa تصبح a2b1c5a3."


@think_for("palindrome")
def _(instr):
    return "المطلوب التحقق مما إذا كانت السلسلة النصية تقرأ بنفس الشكل من الأمام والخلف. الحل الأبسط هو مقارنة السلسلة بمعكوسها باستخدام s == s[::-1] في Python. للسلسلة الفارغة نعيد True. هذه الطريقة بسيطة لكنها تستهلك مساحة إضافية لإنشاء النسخة المعكوسة. بديل أفضل هو استخدام مؤشرين: واحد من البداية وآخر من النهاية ونقارن الحروف مع بعضها حتى يتقاطعا."


@think_for("sliding")
def _(instr):
    return "المطلوب حساب متوسط كل نافذة متصلة بطول k في قائمة أعداد. الفكرة الأساسية هي النافذة المنزلقة: نحسب مجموع النافذة الأولى، ثم عند إزاحة النافذة بمقدار 1 نطرح العنصر الخارج ونضيف العنصر الداخل. هذا يمنع إعادة حساب المجموع من الصفر في كل مرة ويعطي تعقيدا O(n) بدلا من O(n*k). نستخدم قسماة عادية للحصول على متوسطات عشرية."


@think_for("frequent")
def _(instr):
    return "المطلوب إيجاد الحرف الأكثر تكرارا في سلسلة نصية. الفكرة هي استخدام عداد لتسجيل عدد مرات تكرار كل حرف، ثم إيجاد الحرف ذو التكرار الأعلى. نستخدم Counter من collections لتسهيل العد. في حالة وجود أكثر من حرف بنفس التكرار، نختار الحرف الأصغر أبجديا كما هو مطلوب في المسألة."


@think_for("lcm")
def _(instr):
    return "المطلوب حساب المضاعف المشترك الأصغر لعددين. العلاقة الرياضية: LCM(a,b) = a * b / GCD(a,b). لذلك نحتاج أولا حساب GCD باستخدام خوارزمية إقليدس ثم تطبيق العلاقة. يجب استخدام القسمة الصحيحة // لتجنب النتائج العشرية. هذه الطريقة فعالة لأنها تعتمد على حساب GCD ذي التعقيد O(log min(a,b))."


@think_for("binary")
def _(instr):
    return "المطلوب تحويل عدد عشري إلى تمثيله الثنائي. الفكرة: نقسم العدد على 2 ونأخذ الباقي الذي يمثل الرقم الثنائي الأقل أهمية، ثم نكرر على ناتج القسمة حتى يصبح صفرا. نضيف كل باق إلى بداية السلسلة الناتجة. حالة العدد 0 تعيد 0. التعقيد O(log n) لأن عدد مرات القسمة يساوي عدد البتات في التمثيل الثنائي."


@think_for("rotate")
def _(instr):
    return "المطلوب تدوير قائمة إلى اليمين بمقدار k خطوات. الفكرة: تدوير القائمة لليمين يعني أن آخر k عناصر تنتقل إلى البداية. نستخدم عملية التقطيع: nums[-k:] + nums[:-k]. نأخذ k modulo طول القائمة لتجنب التدوير الزائد عن الحاجة. حالة القائمة الفارغة تعيد قائمة فارغة."


@think_for("sum_digits")
def _(instr):
    return "المطلوب حساب مجموع أرقام عدد صحيح باستخدام الاستدعاء الذاتي. الفكرة: نأخذ باقي قسمة العدد على 10 لنحصل على الرقم الأخير، ثم نستدعي الدالة على ناتج القسمة الصحيحة للعدد على 10. الحالة الأساسية هي عندما يصبح العدد صفرا فنعيد 0. مثلا 456: 6 + sum_digits(45) = 6 + 5 + sum_digits(4) = 6 + 5 + 4 = 15."


@think_for("missing_number")
def _(instr):
    return "المطلوب إيجاد العدد المفقود من تسلسل الأعداد من 0 إلى n. الفكرة: مجموع الأعداد من 0 إلى n هو n*(n+1)/2. نحسب مجموع الأعداد الموجودة في القائمة ثم نطرحه من المجموع المتوقع. الفرق هو العدد المفقود. هذه الطريقة تعمل بزمن O(n) ومساحة O(1)، وهي أفضل من تخزين جميع الأرقام في مجموعة."


@think_for("connected_components")
def _(instr):
    return "المطلوب حساب عدد المكونات المتصلة في رسم بياني غير موجه. الفكرة: نمر على جميع العقد، وكلما وجدنا عقدة غير مزارة نبدأ منها DFS أو BFS ونزور جميع العقد المتصلة بها، ثم نزيد العداد. بهذه الطريقة كل DFS يزور مكونا متصلا كاملا. نستخدم مجموعة visited لتتبع العقد المزارة. التعقيد O(V+E) حيث V عدد العقد و E عدد الحواف."


@think_for("greater_element")
def _(instr):
    return "المطلوب إيجاد العنصر الأكبر التالي لكل عنصر في مصفوفة. الفكرة الأساسية هي استخدام مكدس رتيب متناقص: نمر على المصفوفة ونبقي المكدس مرتبا بحيث تكون العناصر فيه متناقصة. لكل عنصر جديد، نزيل من المكدس جميع العناصر الأصغر منه، وهذه العناصر المحذوفة يكون العنصر الحالي هو العنصر الأكبر التالي لها. نضع العنصر الحالي في المكدس ونكمل. العناصر التي تبقى في المكدس بعد نهاية المصفوفة ليس لها عنصر أكبر تالي ونسجل لها 1-."


@think_for("components")
def _(instr):
    return "المطلوب حساب عدد المناطق المتصلة. نستخدم DFS أو BFS لاجتياز كل منطقة والعلام على جميع خلاياها كمزارة. كل منطقة جديدة نكتشفها تزيد العداد بمقدار 1. نمر على جميع الخلايا، وأي خلية غير مزارة نبدأ منها DFS ونوسم المنطقة بأكملها."


@think_for("divisors")
def _(instr):
    return "المطلوب إيجاد جميع القواسم الصحيحة لعدد n. نمر على الأعداد من 1 إلى n ونتحقق مما إذا كان n يقبل القسمة على كل منها بدون باق. الأعداد التي تحقق ذلك هي قواسم n. نضيفها إلى قائمة النتائج. هذه الطريقة البسيطة تعمل بزمن O(n) وهي كافية للمسألة. الأعداد الأولية لها قاسمان فقط: 1 والعدد نفسه."


@think_for("shortest_path")
def _(instr):
    return "المطلوب إيجاد أقصر مسار بين عقدتين في رسم بياني غير موجه. نستخدم BFS لأنه يضمن أقل عدد من الخطوات للوصول لأي عقدة. نبدأ من عقدة البداية بمسافة 0، ونستخدم طابورا. لكل عقدة نسحبها، نفحص جيرانها: إذا كان الجار هو الهدف نعيد المسافة الحالية زائد 1، وإلا نضيفه للطابور إذا لم يزر بعد. إذا انتهى الطابور ولم نصل للهدف نعيد 1-."


@think_for("merge_dicts")
def _(instr):
    return "المطلوب دمج قاموسين مع جمع القيم المتكررة. نبدأ بنسخة من القاموس الأول، ثم نمر على القاموس الثاني: لكل مفتاح نضيف قيمته إلى القيمة الموجودة في النتيجة أو نضيف المفتاح بقيمته إذا لم يكن موجودا. استخدام get مع قيمة افتراضية 0 يبسط الكود ويجعله أكثر أمانا."


@think_for("evaluate")
def _(instr):
    return "المطلوب تقييم تعبير postfix. الفكرة: نستخدم مكدسا لتخزين المعاملات. نمر على الرموز من اليسار إلى اليمين: إذا كان الرمز معاملا نضيفه إلى المكدس، وإذا كان عامل عملية نخرج المعاملين الأخيرين من المكدس ونطبق العملية ثم نعيد النتيجة إلى المكدس. في النهاية، المكدس يحتوي على قيمة التعبير الكامل. الترتيب عند إخراج المعاملين مهم: الأول هو المعامل الأيمن والثاني هو الأيسر."


@think_for("pattern")
def _(instr):
    return "المطلوب التحقق من تطابق نمط مع سلسلة نصية حيث * تطابق أي عدد من الحروف و ? تطابق حرفا واحدا. نستخدم البرمجة الديناميكية: نعرف dp[i][j] بأنه هل يطابق أول i أحرف من النص أول j أحرف من النمط. نبدأ بـ dp[0][0] = True. إذا كان النمط يحوي *، يمكن أن يطابق صفرا أو أكثر من الأحرف. نملأ الجدول تصاعديا ثم نعيد dp[m][n]."


@think_for("power")
def _(instr):
    return "المطلوب حساب قوة عدد: X مرفوع للأس N. نبدأ بنتيجة تساوي 1 ونضربها في X في حلقة تتكرر N مرة. إذا كان N صفرا نعيد 1 مباشرة لأن أي عدد مرفوع للأس صفر يساوي 1. هذه الطريقة البسيطة تعمل بزمن O(N). يمكن تحسينها باستخدام الأس السريع O(log N) لكن الحل البسيط كاف للمسألة."


@think_for("prime")
def _(instr):
    return "المطلوب التحقق مما إذا كان عدد أوليا. نتحقق أولا من الحالات البسيطة: الأعداد أقل من 2 ليست أولية، 2 و 3 أوليان. ثم نتأكد من أن العدد ليس زوجيا. بعد ذلك نفحص القسمة على الأعداد الفردية من 3 حتى الجذر التربيعي للعدد. إذا وجدنا قاسما نعيد False، وإلا فالعدد أولي. نكتفي بالجذر التربيعي لأن أي عامل أكبر من الجذر يقابله عامل أصغر منه."


@think_for("queue")
def _(instr):
    return "المطلوب تنفيذ عمليات على طابور دائري أو بنية بيانات طابور. الفكرة: الطابور يعمل على مبدأ أول دخول أول خروج. نستخدم قائمة من Python أو مكدسين لمحاكاة الطابور. الدالة is_full تتحقق ببساطة مما إذا كان طول الطابور يساوي الحجم الأقصى المحدد."


@think_for("sort")
def _(instr):
    return "المطلوب ترتيب عناصر وفق معيار محدد. نستخدم دالة sorted المدمجة مع معيار key المخصص حسب المطلوب: الترتيب حسب الطول أو حسب قيمة في قاموس أو حسب معيار متعدد مثل الدرجة تنازليا ثم الاسم أبجديا."


@think_for("reversal")
def _(instr):
    return "المطلوب عكس قائمة أو مكدس. الفكرة: لعكس مكدس باستخدام الاستدعاء الذاتي، نزيل العنصر العلوي، نعكس باقي المكدس، ثم ندرج العنصر المحذوف في القاعدة. هذه التقنية تستدعي نفسها حتى يصبح المكدس فارغا ثم تبدأ في إعادة بناء المكدس معكوسا."


# Default fallback thinks
DEFAULT_THINKS = {
    "كتابة دالة": "المطلوب كتابة دالة تقوم بحساب أو معالجة البيانات وفق الوصف المطلوب. نحدد أولا المعطيات والنتيجة المرجوة، ثم نختار الخوارزمية المناسبة لحل المسألة مع مراعاة الحالات الحدية مثل القيم الفارغة أو الأعداد السالبة أو الفهارس خارج النطاق.",
    "إيجاد الخطأ": "المطلوب إيجاد الخطأ في الكود المقدم وتصحيحه. نفحص الكود سطرا سطرا ونبحث عن الأخطاء الشائعة مثل: عدم معالجة الحالات الحدية، أخطاء في شروط الحلقات، استخدام متغيرات غير معرفة، أو منطق خاطئ في العمليات الحسابية أو الشرطية.",
    "تحسين الكفاءة": "المطلوب تحسين كفاءة الكود المقدم. نحدد أولا سبب البطء: هل هو تكرار غير ضروري، حلقات متداخلة، أو استخدام خوارزمية غير مناسبة؟ ثم نقدم حلا محسنا باستخدام بنى بيانات أكثر كفاءة كالمجموعات والقواميس أو خوارزميات أسرع مثل النافذة المنزلقة أو البرمجة الديناميكية.",
    "إكمال الكود": "المطلوب إكمال الجزء الناقص من الكود. نفهم أولا السياق العام للكود ونحدد الجزء المطلوب إكماله بناء على اسم الدالة والمعطيات والنتيجة المتوقعة، ثم نكتب الكود المناسب الذي يتكامل مع بقية الأجزاء.",
}


def func_name(instr):
    m = re.search(r'`(\w+)`', instr)
    if m: return m.group(1)
    m = re.search(r'دالة\s+(\w+)', instr)
    if m: return m.group(1)
    return None


def main():
    raw = load_raw()
    final = load_final()
    changes = 0

    for oid in sorted(final.keys()):
        obj = final[oid]
        think = obj.get("think", "")
        answer = obj.get("answer", "")
        is_placeholder = len(answer) < 30 or "pass" in answer
        think_short = len(think.split()) < 60

        if not is_placeholder and not think_short:
            continue

        raw_entry = raw.get(oid)
        if not raw_entry:
            continue

        instr = raw_entry.get("instruction", "")
        topic = raw_entry.get("topic", "")
        ptype = raw_entry.get("problem_type", "")
        ut = raw_entry.get("unit_tests", [])

        fname = func_name(instr)
        new_answer = None
        new_think = None

        if fname and fname in ANSWERS:
            new_answer = ANSWERS[fname](instr)

        if fname:
            for key, fn in THINKS.items():
                if key in fname.lower() or fname.lower().startswith(key):
                    new_think = fn(instr)
                    break

        if not new_think:
            instr_lower = instr.lower()
            for key, fn in THINKS.items():
                if key in instr_lower:
                    new_think = fn(instr)
                    break

        if not new_think:
            new_think = DEFAULT_THINKS.get(ptype, DEFAULT_THINKS["كتابة دالة"])

        if new_answer and (is_placeholder or (len(answer) < 50)):
            obj["answer"] = new_answer
            changes += 1

        if new_think and think_short:
            obj["think"] = new_think
            changes += 1

        obj["unit_tests"] = ut

    with open(FINAL, "w", encoding="utf-8") as f:
        for oid in sorted(final.keys()):
            f.write(json.dumps(final[oid], ensure_ascii=False) + "\n")

    print(f"Updated entries: {changes}")


if __name__ == "__main__":
    main()
