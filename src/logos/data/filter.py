import re

def is_valid_reasoning(think_text):
    if not think_text or len(think_text.split()) < 20:
        return False
    if '###' in think_text or '**' in think_text:
        return False
    
    # Arabic detector (simple check for Arabic characters)
    if not re.search(r'[\u0600-\u06FF]', think_text):
        return False

    lazy_keywords = ["مكرر", "Auto-generated", "حل الخوارزمية", "TODO"]
    for kw in lazy_keywords:
        if kw in think_text:
            return False
    return True
