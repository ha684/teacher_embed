import google.generativeai as genai
import time
import random

# Cấu hình khóa API
GOOGLE_API_KEY = 'AIzaSyDXzLE-DWk1JOOPL9gx0d3fjFoOvryZvAM'
genai.configure(api_key=GOOGLE_API_KEY)

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    safety_settings=SAFETY_SETTINGS,
)

def generate_questions_for_topic(topic, num_questions=10):
    prompt = (
        f"Hãy tạo {num_questions} câu hỏi đa dạng về chủ đề {topic}. Làm cho chúng thú vị và kích thích tư duy. "
        "Chỉ cung cấp câu hỏi, mỗi câu một dòng, không đánh số hoặc dấu đầu dòng."
    )
    try:
        response = model.generate_content(prompt)
        questions = response.text.strip().split('\n')
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        print(f"Lỗi khi tạo câu hỏi về {topic}: {e}")
        return []

def write_questions_to_file(questions, filename="/home/visssoft/workspace/teacher_embed/query.txt"):
    """Ghi các câu hỏi vào tệp, mỗi câu một dòng."""
    with open(filename, 'a', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')

def main():
    topics = [
    # Khoa học và công nghệ
    "khoa học", "công nghệ", "vật lý lượng tử", "robot học", "thực tế ảo", 
    "thực tế tăng cường (AR)", "công nghệ nano", "trí tuệ nhân tạo", "an ninh mạng", 
    "dữ liệu lớn", "công nghệ blockchain", "internet vạn vật (IoT)", "khoa học vật liệu", 
    "năng lượng tái tạo", "kỹ thuật không gian", "vũ khí công nghệ cao", "chế tạo xe tự hành", 

    # Khoa học tự nhiên
    "toán học", "sinh học tiến hóa", "vật lý thiên văn", "hóa sinh học", 
    "địa chất học", "thiên văn học", "khí hậu học", "động vật học", 
    "thực vật học", "hải dương học", "khoa học về đại dương", 
    "công nghệ sinh học", "khoa học vũ trụ", "địa lý vật lý", 

    # Khoa học xã hội và nhân văn
    "lịch sử", "triết học", "ngôn ngữ học", "xã hội học", "tâm lý học hành vi", 
    "kinh tế học vĩ mô", "chính trị quốc tế", "nhân học văn hóa", 
    "văn hóa học so sánh", "giáo dục học hiện đại", "nghiên cứu phát triển", 
    "tâm lý học xã hội", "pháp luật dân sự", "công nghệ giáo dục",

    # Nghệ thuật và giải trí
    "nghệ thuật đương đại", "văn học hiện đại", "âm nhạc điện tử", "điện ảnh tài liệu", 
    "thiết kế thời trang", "thiết kế nội thất", "kiến trúc xanh", 
    "nhiếp ảnh nghệ thuật", "phim hoạt hình", "truyện tranh", "thiết kế nhân vật",

    # Sức khỏe và đời sống
    "y học", "khoa học thần kinh", "dinh dưỡng học", "chăm sóc sức khỏe tâm thần", 
    "tâm lý học tích cực", "vấn đề sức khỏe cộng đồng", "thể thao mạo hiểm", 
    "công nghệ y tế", "liệu pháp gene", "nghiên cứu về tuổi thọ con người",

    # Vấn đề xã hội
    "biến đổi khí hậu", "bất bình đẳng kinh tế", "quyền lợi lao động", "xóa đói giảm nghèo",
    "vấn đề giáo dục trẻ em", "chống bạo lực gia đình", "bình đẳng sắc tộc", 
    "quyền của người khuyết tật", "phát triển nông thôn", "di cư do khí hậu", 

    # Kỹ năng và phát triển cá nhân
    "kỹ năng viết sáng tạo", "quản lý dự án", "xây dựng thương hiệu cá nhân", 
    "kỹ năng tư duy phản biện", "kỹ năng đàm phán", "tư duy sáng tạo", 
    "phát triển lãnh đạo", "ứng dụng mindfulness", "kỹ năng học tập hiệu quả", 
    "lập kế hoạch chiến lược cá nhân", 

    # Các chủ đề khác
    "khám phá ẩm thực", "nghiên cứu tôn giáo", "hòa bình và xung đột", 
    "văn hóa đại chúng", "giải mã giấc mơ", "phân tích thị trường chứng khoán", 
    "khám phá hành tinh ngoài hệ mặt trời", "lịch sử âm nhạc cổ điển", 
    "di sản văn hóa thế giới", "nghệ thuật graffiti", "nghiên cứu về thế giới tương lai", 
    "phát triển game", "khoa học về thói quen"
]


    total_questions = 0

    print("Bắt đầu tạo câu hỏi...")

    while True:
        topic = random.choice(topics)
        print(f"Đang tạo câu hỏi về chủ đề {topic}...")
        questions = generate_questions_for_topic(topic)
        write_questions_to_file(questions)
        total_questions += len(questions)
        print(f"Đã tạo {len(questions)} câu hỏi về chủ đề {topic}")

        time.sleep(2)

    print(f"\nHoàn tất! Đã tạo tổng cộng {total_questions} câu hỏi.")
    print("Các câu hỏi đã được lưu vào tệp 'generated_questions_vn.txt'")

if __name__ == "__main__":
    main()
