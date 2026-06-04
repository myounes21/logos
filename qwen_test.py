from openai import OpenAI

from config import settings


def main() -> None:
    client = OpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_BASE_URL,
    )

    prompt = (
        "أنت مولّد مسائل برمجية باللغة العربية.\n\n"
        "مهمتك: توليد 10 مسألة برمجية بالعربية بصيغة JSONL "
        "(كل مسألة في سطر JSON مستقل).\n\n"
        "المواصفات المطلوبة لكل مسألة:\n\n"
        "1. topic: \"مؤشرين\" | subtopic: \"تقسيم قائمة\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "2. topic: \"استدعاء ذاتي\" | subtopic: \"توليد التباديل\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "3. topic: \"نافذة منزلقة\" | subtopic: \"تكرار الأحرف\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "4. topic: \"برمجة ديناميكية\" | subtopic: \"Grid Paths\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "5. topic: \"استدعاء ذاتي\" | subtopic: \"Factorial\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "6. topic: \"مؤشرين\" | subtopic: \"Pair Sum\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "7. topic: \"ترتيب\" | subtopic: \"ترتيب جزئي\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "8. topic: \"جشع\" | subtopic: \"Minimum Operations\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "9. topic: \"رسوم بيانية\" | subtopic: \"Shortest Path\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "10. topic: \"نافذة منزلقة\" | subtopic: \"Sliding Average\" | "
        "difficulty: \"سهل\" | problem_type: \"كتابة دالة\"\n\n"
        "قواعد صارمة:\n\n"
        "1. الـ instruction يجب أن يكون بالعربية الفصحى بالكامل\n\n"
        "2. لا تكتب الحل — فقط المسألة والاختبارات\n\n"
        "3. unit_tests: قائمة من 3 إلى 5 اختبارات تشمل حالات حدية "
        "(edge cases)\n\n"
        "4. كل اختبار بصيغة: {\"input\": \"...\", \"expected\": \"...\"}\n\n"
        "5. إذا كان problem_type هو \"إيجاد الخطأ\": أدرج كوداً "
        "بايثون خاطئاً داخل الـ instruction\n\n"
        "6. إذا كان problem_type هو \"تحسين الكفاءة\": أدرج كوداً "
        "بايثون يعمل لكنه بطيء داخل الـ instruction\n\n"
        "7. الرد يجب أن يكون JSONL فقط — لا شرح، لا مقدمة، لا نص خارج الـ JSON\n\n"
        "صيغة كل سطر:\n\n"
        "{\"instruction\": \"...\", \"topic\": \"...\", "
        "\"subtopic\": \"...\", \"difficulty\": \"...\", "
        "\"problem_type\": \"...\", \"unit_tests\": [...]}\n\n"
        "ابدأ الآن:"
    )

    response = client.chat.completions.create(
        model=settings.QWEN_MODEL,
        messages=[
            {"role": "system", "content": "أنت مولّد مسائل برمجية باللغة العربية."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    message = response.choices[0].message
    print(message.content or "(no content)")


if __name__ == "__main__":
    main()
