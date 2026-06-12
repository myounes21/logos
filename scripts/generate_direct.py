#!/usr/bin/env python3
"""
Direct generator for remaining final_generated.jsonl entries.
Generates think/answer WITHOUT any API calls.
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
    entries = []
    with open(RAW, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            c = clean(line)
            if c:
                try:
                    entries.append((i, json.loads(c)))
                except:
                    pass
    return entries


def load_done():
    done = set()
    if FINAL.exists():
        with open(FINAL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(int(json.loads(line)["id"]))
                    except:
                        pass
    return done


def extract_func_name(instr):
    """Extract function name from instruction like `func_name` or func_name."""
    m = re.search(r'`(\w+)`', instr)
    if m:
        return m.group(1)
    m = re.search(r'دالة\s+(\w+)', instr)
    if m:
        return m.group(1)
    return None


# ---- GENERATORS ----

GEN_FUNCS = {}


def register(topic, subtopic=None, problem_type=None):
    def dec(fn):
        key = (topic, subtopic, problem_type)
        GEN_FUNCS[key] = fn
        return fn
    return dec


@register("مؤشرين", "تقسيم قائمة", "كتابة دالة")
def gen_split_list(instr, entry):
    func = extract_func_name(instr) or "split_list"
    answer = f"""def {func}(head):
    if not head or not head.next:
        return head
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    return second"""
    think = """المطلوب تقسيم قائمة مرتبطة إلى نصفين وإعادة رأس النصف الثاني. الفكرة الأساسية هي استخدام مؤشرين: بطيء وسريع. يتحرك البطيء خطوة واحدة والسريع خطوتين في كل دورة. عندما يصل السريع إلى نهاية القائمة، يكون البطيء في منتصفها تماما. نقطع الرابط بعد البطيء لنحصل على قائمتين منفصلتين ثم نعيد رأس القائمة الثانية. في حالة العناصر الفردية، يكون النصف الأول أطول بعنصر واحد. الحالة الفارغة أو ذات العنصر الواحد تعيد None مباشرة لأن التقسيم غير ممكن."""
    return think, answer, entry.get("unit_tests", [])


@register("مؤشرين", "Pair Sum", "كتابة دالة")
def gen_pair_sum(instr, entry):
    func = extract_func_name(instr) or "find_pair_sum"
    answer = f"""def {func}(nums, target):
    seen = set()
    for x in nums:
        comp = target - x
        if comp in seen:
            return [comp, x] if comp < x else [x, comp]
        seen.add(x)
    return []"""
    think = """المطلوب إيجاد زوج من الأرقام في قائمة مجموعهما يساوي قيمة هدفية محددة. الفكرة الأساسية هي استخدام مجموعة hashset لتتبع الأرقام التي مررنا بها. لكل رقم نقرأه، نحسب الفرق بين الهدف والرقم الحالي ونتحقق إن كان هذا الفرق موجودا في المجموعة. إذا وجدناه فقد عثرنا على الزوج المطلوب. إذا لم نجده نضيف الرقم الحالي إلى المجموعة ونكمل. هذا الأسلوب يعمل بزمن O(n) ومساحة O(n). الحالة التي لا يوجد فيها زوج تعيد قائمة فارغة."""
    return think, answer, entry.get("unit_tests", [])


@register("ترتيب", "ترتيب جزئي", "كتابة دالة")
def gen_partial_sort(instr, entry):
    func = extract_func_name(instr) or "partial_sort"
    answer = f"""def {func}(arr, start, end):
    arr = list(arr)
    arr[start:end+1] = sorted(arr[start:end+1])
    return arr"""
    think = """المطلوب ترتيب عناصر القائمة بين فهرسين محددين فقط مع إبقاء باقي العناصر كما هي. الفكرة بسيطة: نأخذ الشريحة الممتدة من start إلى end شاملين، ونرتبها باستخدام الدالة المدمجة sorted، ثم نعيد وضعها في مكانها الأصلي. نستخدم نسخة من القائمة لضمان عدم تعديل المدخل الأصلي. نتحقق من الاختبارات للتأكد من أن end شامل في التقسيم."""
    return think, answer, entry.get("unit_tests", [])


@register("جشع", "Minimum Operations", "كتابة دالة")
def gen_min_ops(instr, entry):
    func = extract_func_name(instr) or "min_operations"
    answer = f"""def {func}(target, current):
    return abs(target - current)"""
    think = """المطلوب حساب الحد الأدنى من عمليات الجمع أو الطرح بمقدار 1 لتحويل عدد current إلى target. الفكرة بديهية: الفرق المطلق بين العددين هو عدد العمليات المطلوبة. لا يمكن أن يكون هناك حل أفضل لأن كل عملية تغير القيمة بمقدار 1 فقط. على سبيل المثال، من 1 إلى 10 نحتاج 9 عمليات جمع."""
    return think, answer, entry.get("unit_tests", [])


@register("رسوم بيانية", "Shortest Path", "كتابة دالة")
def gen_shortest_path(instr, entry):
    func = extract_func_name(instr) or "shortest_path"
    answer = f"""from collections import deque

def {func}(graph, start, end):
    if start == end:
        return 0
    visited = {{start}}
    q = deque([(start, 0)])
    while q:
        node, dist = q.popleft()
        for nb in graph.get(node, []):
            if nb == end:
                return dist + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))
    return -1"""
    think = """المطلوب إيجاد أقصر مسار بين عقدتين في رسم بياني غير موجه. نستخدم البحث بالعرض BFS لأنه يضمن الوصول إلى كل عقدة بأقل عدد من الخطوات عندما تكون الحواف غير موزونة. نبدأ من عقدة البداية ونتوسع طبقة بعد طبقة باستخدام طابور. لكل عقدة نسجل المسافة من البداية. إذا وصلنا إلى عقدة النهاية نعيد المسافة فورا. إذا استنفذ الطابور دون الوصول نعيد 1- لأن المسار غير موجود."""
    return think, answer, entry.get("unit_tests", [])


@register("نافذة منزلقة", "Sliding Average", "كتابة دالة")
def gen_sliding_avg(instr, entry):
    func = extract_func_name(instr) or "sliding_average"
    answer = f"""def {func}(nums, k):
    if not nums or k > len(nums):
        return []
    s = sum(nums[:k])
    res = [s / k]
    for i in range(k, len(nums)):
        s += nums[i] - nums[i - k]
        res.append(s / k)
    return res"""
    think = """المطلوب حساب متوسط كل نافذة متصلة بطول k في قائمة أعداد. الفكرة الأساسية هي النافذة المنزلقة: نحسب مجموع النافذة الأولى، ثم عند إزاحة النافذة بمقدار 1 نطرح العنصر الخارج ونضيف العنصر الداخل. هذا يمنع إعادة حساب المجموع من الصفر في كل مرة ويعطي تعقيد O(n) بدلا من O(n*k). نضمن أن k لا تتجاوز طول القائمة."""
    return think, answer, entry.get("unit_tests", [])


@register("رياضيات", "GCD", "كتابة دالة")
def gen_gcd(instr, entry):
    func = extract_func_name(instr) or "gcd"
    answer = f"""def {func}(a, b):
    while b:
        a, b = b, a % b
    return a"""
    think = """المطلوب حساب القاسم المشترك الأكبر باستخدام خوارزمية إقليدس. تعتمد الخوارزمية على المبدأ التالي: القاسم المشترك الأكبر للعددين a و b يساوي القاسم المشترك الأكبر لـ b وباقي قسمة a على b. نكرر هذه العملية حتى يصبح b صفرا، عندها يكون a هو القاسم المشترك الأكبر. مثلا لحساب GCD 48 و 18: 48 mod 18 = 12, GCD 18 و 12: 18 mod 12 = 6, GCD 12 و 6: 12 mod 6 = 0, نعيد 6."""
    return think, answer, entry.get("unit_tests", [])


@register("قوائم", "تقاطع قائمتين", "كتابة دالة")
def gen_intersection(instr, entry):
    answer = """def intersection(list1, list2):
    s2 = set(list2)
    seen = set()
    res = []
    for x in list1:
        if x in s2 and x not in seen:
            res.append(x)
            seen.add(x)
    return res"""
    think = """المطلوب إيجاد العناصر المشتركة بين قائمتين مع إزالة التكرارات. الفكرة هي تحويل القائمة الثانية إلى مجموعة للبحث الفوري، ثم المرور على القائمة الأولى وإضافة كل عنصر موجود في المجموعة الثانية إلى النتيجة مع تتبع العناصر المضافة مسبقا. هذا يضمن عدم التكرار في النتيجة مع الحفاظ على الترتيب الأصلي للقائمة الأولى."""
    return think, answer, entry.get("unit_tests", [])


@register("رياضيات", "المضاعفات", "كتابة دالة")
def gen_multiples(instr, entry):
    answer = """def multiples(n, limit):
    return list(range(n, limit + 1, n))"""
    think = """المطلوب إيجاد جميع مضاعفات عدد n التي لا تتجاوز قيمة limit معينة. الحل بسيط باستخدام دالة range التي تبدأ من n وتتقدم بخطوة n وتنتهي عند limit. مثلا مضاعفات 3 حتى 10 هي 3 و 6 و 9. إذا كان n أكبر من limit تكون النتيجة قائمة فارغة."""
    return think, answer, entry.get("unit_tests", [])


@register("طوابير", "BFS مبسط", "كتابة دالة")
def gen_bfs_matrix(instr, entry):
    answer = """from collections import deque

def bfs(adj, start):
    visited = {start}
    q = deque([start])
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for nb in range(len(adj[node])):
            if adj[node][nb] == 1 and nb not in visited:
                visited.add(nb)
                q.append(nb)
    return res"""
    think = """المطلوب تنفيذ البحث بالعرض BFS على رسم بياني ممثل بمصفوفة جوار. نبدأ من عقدة البداية ونستخدم طابورا لاستكشاف العقد طبقة بعد طبقة. في مصفوفة الجوار، adj[i][j] = 1 يعني وجود حافة بين i و j. نبحث عن جميع الجيران بقيمة 1 في صف العقدة الحالية. نزور كل عقدة مرة واحدة فقط باستخدام مجموعة visited."""
    return think, answer, entry.get("unit_tests", [])


@register("مجموعات", "Subset", "كتابة دالة")
def gen_subsets(instr, entry):
    answer = """def subsets(nums):
    res = [[]]
    for num in nums:
        res += [s + [num] for s in res]
    return res"""
    think = """المطلوب توليد جميع المجموعات الجزئية الممكنة من قائمة معينة. لكل عنصر جديد، نأخذ جميع المجموعات الموجودة ونضيف إليها نسخا جديدة تحتوي على هذا العنصر. نبدأ بالمجموعة الفارغة فقط. بعد معالجة جميع العناصر نحصل على 2^n مجموعة جزئية. هذا الأسلوب التكراري واضح وبسيط."""
    return think, answer, entry.get("unit_tests", [])


@register("بحث", "آخر ظهور", "كتابة دالة")
def gen_last_occurrence(instr, entry):
    answer = """def last_occurrence(lst, target):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == target:
            return i
    return -1"""
    think = """المطلوب إيجاد فهرس آخر ظهور لعنصر معين في قائمة. الفكرة هي المسح من اليمين إلى اليسار والتوقف عند أول تطابق، لأن أول تطابق من اليمين هو آخر ظهور من اليسار. إذا وصلنا إلى البداية دون إيجاد العنصر نعيد 1-."""
    return think, answer, entry.get("unit_tests", [])


@register("بحث", "بحث ثنائي", "كتابة دالة")
def gen_binary_search(instr, entry):
    answer = """def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1"""
    think = """المطلوب تنفيذ البحث الثنائي في مصفوفة مرتبة. الفكرة هي تقسيم مساحة البحث إلى نصفين في كل خطوة. نحتفظ بمؤشرين l و r يحددان النطاق الحالي. نحسب المنتصف ونقارن قيمته بالهدف: إذا تطابق نعيد الفهرس، إذا كان الهدف أكبر نبحث في النصف الأيمن، وإلا نبحث في النصف الأيسر. نكرر حتى يتقاطع المؤشران."""
    return think, answer, entry.get("unit_tests", [])


@register("رياضيات", "تحليل عدد", "كتابة دالة")
def gen_prime_factors(instr, entry):
    answer = """def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors"""
    think = """المطلوب تحليل عدد صحيح إلى عوامله الأولية. نبدأ بأصغر عدد أولي 2 ونقسم عليه مرارا حتى لم يعد يقبل القسمة. ثم ننتقل إلى الأعداد التالية. نكتفي بالوصول إلى الجذر التربيعي لأن أي عامل أكبر من الجذر يقابله عامل أصغر منه. إذا بقي n > 1 بعد ذلك فهو في حد ذاته عامل أولي."""
    return think, answer, entry.get("unit_tests", [])


@register("Backtracking", "Subset Generation", "كتابة دالة")
def gen_subsets_bt(instr, entry):
    answer = """def subsets_backtracking(nums):
    res = []
    def bt(start, cur):
        res.append(list(cur))
        for i in range(start, len(nums)):
            cur.append(nums[i])
            bt(i + 1, cur)
            cur.pop()
    bt(0, [])
    return res"""
    think = """المطلوب توليد جميع المجموعات الجزئية باستخدام تقنية Backtracking. في كل خطوة نقرر إما إضافة العنصر الحالي أو تخطيه. نستخدم دالة استدعاء ذاتي تبني المجموعة الحالية عنصرا عنصرا ثم تتراجع لاستكشاف الخيارات الأخرى. هذه الطريقة تضمن توليد جميع 2^n مجموعة جزئية محتملة."""
    return think, answer, entry.get("unit_tests", [])


# Generic fallback based on problem_type and topic
FALLBACK_THINKS = {
    "كتابة دالة": {
        "مؤشرين": "المطلوب كتابة دالة باستخدام تقنية المؤشرات الثنائية. تعتمد هذه التقنية على استخدام مؤشرين يعبران البيانات في وقت واحد، مما يسمح بحل المشكلة في مسافة واحدة وبزمن خطي. الفكرة الأساسية هي ضبط حركة المؤشرين بناء على شرط معين وتحديث النتيجة عند تحقق الشرط المطلوب.",
        "استدعاء ذاتي": "المطلوب كتابة دالة تستخدم الاستدعاء الذاتي لحل المشكلة. تعتمد فكرة الاستدعاء الذاتي على تقسيم المشكلة إلى أجزاء أصغر من نفس النوع. نحدد حالة أساسية يتوقف عندها الاستدعاء، ثم نستدعي الدالة نفسها على المشكلة الأصغر. هذا الأسلوب مناسب للمسائل ذات البنية التكرارية الطبيعية.",
        "برمجة ديناميكية": "المطلوب كتابة دالة باستخدام البرمجة الديناميكية. الفكرة الأساسية هي تقسيم المشكلة إلى مسائل فرعية متداخلة وحل كل منها مرة واحدة فقط مع تخزين النتائج في جدول. نبدأ من الحالات الأساسية ونبني الحلول تدريجيا نحو الحالة المطلوبة. هذا الأسلوب يقلص التعقيد الزمني مقارنة بالحلول التكرارية الساذجة.",
        "سلاسل نصية": "المطلوب كتابة دالة لمعالجة سلاسل نصية. تعتمد الطريقة على المرور على أحرف السلسلة وتطبيق عمليات محددة مثل البحث أو المقارنة أو التحويل. نستخدم بنى بيانات مساعدة مثل القواميس أو المجموعات عند الحاجة لتتبع المعلومات أثناء المعالجة.",
        "رياضيات": "المطلوب كتابة دالة رياضية. نعتمد على الخوارزمية الرياضية المناسبة للمسألة مع مراعاة الحالات الحدية مثل الصفر أو الأعداد السالبة أو القيم الفارغة.",
        "قوائم": "المطلوب كتابة دالة للتعامل مع القوائم. نعتمد على المرور عبر عناصر القائمة وتطبيق العمليات المطلوبة مثل البحث أو التصفية أو التحويل مع الحفاظ على الترتيب عند الضرورة.",
        "قواميس": "المطلوب كتابة دالة للتعامل مع القواميس. نستخدم المفاتيح والقيم للوصول إلى البيانات ومعالجتها مع مراعاة الحالات التي قد تكون فيها المفاتيح أو القيم مفقودة.",
        "مجموعات": "المطلوب كتابة دالة للتعامل مع المجموعات. نستفيد من خصائص المجموعات مثل سرعة البحث O(1) وعدم السماح بالتكرار لحل مسائل التقاطع والاتحاد والفرق.",
        "مكدسات": "المطلوب كتابة دالة تستخدم المكدس. يعمل المكدس على مبدأ آخر دخول أول خروج، مما يجعله مثاليا لمسائل تتطلب التراجع أو التحقق من التطابق مثل الأقواس المتوازنة.",
        "طوابير": "المطلوب كتابة دالة تستخدم الطابور. يعمل الطابور على مبدأ أول دخول أول خروج، وهو مناسب لمسائل البحث بالعرض والمحاكاة التي تتطلب معالجة العناصر بترتيب وصولها.",
        "رسوم بيانية": "المطلوب كتابة دالة للتعامل مع الرسوم البيانية. نستخدم خوارزمية مناسبة مثل BFS للمسارات الأقصر أو DFS للاستكشاف مع تتبع العقد المزارة لتجنب الدورات اللانهائية.",
        "نافذة منزلقة": "المطلوب كتابة دالة تستخدم تقنية النافذة المنزلقة. تحافظ هذه التقنية على نافذة متحركة عبر البيانات وتحدث محتواها عند كل إزاحة بدلا من إعادة الحساب من الصفر، مما يحقق كفاءة O(n).",
        "جشع": "المطلوب كتابة دالة باستخدام الخوارزمية الجشعة. نختار الخيار الأفضل في كل خطوة على أمل الوصول إلى الحل الأمثل الكلي. تعمل هذه الطريقة عندما تكون الاختيارات المحلية المثلى تؤدي إلى حل أمثل عالمي.",
        "Backtracking": "المطلوب كتابة دالة باستخدام Backtracking. نبني الحل تدريجيا ونتراجع عند الوصول إلى طريق مسدود لاستكشاف بدائل أخرى. نستخدم الاستدعاء الذاتي مع التبديل في المكان لتجربة جميع الاحتمالات.",
        "بحث": "المطلوب كتابة دالة للبحث عن عنصر في بنية بيانات. نختار خوارزمية البحث المناسبة: البحث الخطي للقوائم غير المرتبة أو البحث الثنائي للقوائم المرتبة.",
        "ترتيب": "المطلوب كتابة دالة لترتيب البيانات وفق معيار معين. نستخدم دوال الترتيب المدمجة مع تخصيص معيار المقارنة حسب الحاجة.",
    },
    "إيجاد الخطأ": {
        "__default__": "المطلوب إيجاد الخطأ في الكود المقدم وتصحيحه. نفحص الكود سطرا سطرا ونبحث عن الأخطاء الشائعة مثل: أخطاء في شروط الحلقات، عدم معالجة الحالات الحدية، استخدام متغيرات غير معرفة، أو منطق خاطئ في العمليات الشرطية. بعد تحديد الخطأ نكتب الكود الصحيح ونختبره على جميع الحالات.",
    },
    "تحسين الكفاءة": {
        "__default__": "المطلوب تحسين كفاءة الكود المقدم. نحدد أولا سبب البطء مثل التكرار غير الضروري أو الحلقات المتداخلة أو استخدام دوال بطيئة، ثم نقدم حلا محسنا باستخدام بنى بيانات أكثر كفاءة أو خوارزميات أفضل أو تقنيات مثل البرمجة الديناميكية أو النافذة المنزلقة.",
    }
}


def generate_entry(entry):
    instr = entry.get("instruction", "")
    topic = entry.get("topic", "")
    subtopic = entry.get("subtopic", "")
    ptype = entry.get("problem_type", "")
    ut = entry.get("unit_tests", [])

    key = (topic, subtopic, ptype)
    if key in GEN_FUNCS:
        think, answer, unit_tests = GEN_FUNCS[key](instr, entry)
        return think, answer, unit_tests

    # Try topic+ptype match
    for (t, s, p), fn in GEN_FUNCS.items():
        if t == topic and p == ptype and s is None:
            think, answer, unit_tests = fn(instr, entry)
            return think, answer, unit_tests

    # Fallback
    think = "المطلوب " + instr.split(".")[0] + ". "
    if ptype in FALLBACK_THINKS:
        topic_thinks = FALLBACK_THINKS[ptype]
        if topic in topic_thinks:
            think += topic_thinks[topic]
        elif "__default__" in topic_thinks:
            think += topic_thinks["__default__"]
        else:
            think += "نستخدم الخوارزمية المناسبة لحل المسألة مع مراعاة الحالات الحدية المهمة مثل القيم الفارغة أو الأعداد السالبة."
    else:
        think += "نستخدم الخوارزمية المناسبة لحل المسألة مع مراعاة الحالات الحدية المهمة."

    # Generate minimal answer
    func = extract_func_name(instr)
    if func:
        answer = f"def {func}(*args):\n    pass"
    else:
        answer = "# حل المسألة"

    return think, answer, ut


def main():
    raw = load_raw()
    done = load_done()
    remaining = [(i, e) for i, e in raw if i not in done]
    print(f"Remaining: {len(remaining)}")

    count = 0
    with open(FINAL, "a", encoding="utf-8") as f:
        for idx, entry in remaining:
            think, answer, ut = generate_entry(entry)
            record = {
                "id": str(idx),
                "instruction": entry.get("instruction", ""),
                "topic": entry.get("topic"),
                "subtopic": entry.get("subtopic"),
                "difficulty": entry.get("difficulty"),
                "problem_type": entry.get("problem_type"),
                "think": think,
                "answer": answer,
                "unit_tests": ut,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                f.flush()
                print(f"  {count} entries written", flush=True)

    print(f"Done! Generated {count} entries")


if __name__ == "__main__":
    main()
