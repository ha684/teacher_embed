import os
import requests
from tqdm import tqdm
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

random.seed(42)

queries = [
    "Pháp luật Việt Nam quy định như thế nào về quyền sở hữu trí tuệ?",
    "Làm thế nào để đăng ký sở hữu trí tuệ tại Việt Nam?",
    "Các khoản thuế phải nộp khi mua bán bất động sản là gì?",
    "Lãi suất tiết kiệm hiện nay là bao nhiêu?",
    "Làm thế nào để đăng ký học trực tuyến tại một trường đại học?",
    "Các quy định về bảo hiểm xã hội cho người lao động là gì?",
    "Làm thế nào để mở tài khoản ngân hàng trực tuyến?",
    "Các khoản phí khi vay tiền mua nhà là gì?",
    "Pháp luật quy định như thế nào về quyền tự do ngôn luận?",
    "Làm thế nào để đăng ký kinh doanh tại Việt Nam?",
    "Các khoản phí khi gửi tiền quốc tế là gì?",
    "Làm thế nào để đăng ký thi chứng chỉ tiếng Anh TOEIC?",
    "Các quy định về bảo hiểm y tế cho học sinh là gì?",
    "Làm thế nào để tính lãi suất kép trên tài khoản tiết kiệm?",
    "Pháp luật quy định như thế nào về quyền sở hữu nhà ở?",
    "Làm thế nào để đăng ký học phần tại trường đại học?",
    "Các khoản phí khi mở tài khoản chứng khoán là gì?",
    "Làm thế nào để đăng ký bảo hiểm xe máy?",
    "Các quy định về bảo hiểm thất nghiệp là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư chứng khoán?",
    "Pháp luật quy định như thế nào về quyền tố cáo?",
    "Làm thế nào để đăng ký thi chứng chỉ CPA?",
    "Các khoản phí khi gửi tiền qua ngân hàng là gì?",
    "Làm thế nào để đăng ký học tiếng Anh tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hóa là gì?",
    "Làm thế nào để tính lãi suất vay ngân hàng?",
    "Pháp luật quy định như thế nào về quyền sở hữu đất đai?",
    "Làm thế nào để đăng ký thi chứng chỉ CFA?",
    "Các khoản phí khi mở tài khoản tiết kiệm là gì?",
    "Làm thế nào để đăng ký bảo hiểm nhân thọ?",
    "Các quy định về bảo hiểm tai nạn lao động là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư bất động sản?",
    "Pháp luật quy định như thế nào về quyền bầu cử?",
    "Làm thế nào để đăng ký thi chứng chỉ ACCA?",
    "Các khoản phí khi gửi tiền qua ví điện tử là gì?",
    "Làm thế nào để đăng ký học lập trình tại một trung tâm?",
    "Các quy định về bảo hiểm du lịch là gì?",
    "Làm thế nào để tính lãi suất tiền gửi ngân hàng?",
    "Pháp luật quy định như thế nào về quyền sở hữu công nghiệp?",
    "Làm thế nào để đăng ký thi chứng chỉ CMA?",
    "Các khoản phí khi mở tài khoản ngân hàng là gì?",
    "Làm thế nào để đăng ký bảo hiểm ô tô?",
    "Các quy định về bảo hiểm sức khỏe là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư trái phiếu?",
    "Pháp luật quy định như thế nào về quyền khiếu nại?",
    "Làm thế nào để đăng ký thi chứng chỉ CISA?",
    "Các khoản phí khi gửi tiền qua bưu điện là gì?",
    "Làm thế nào để đăng ký học nghề tại một trung tâm?",
    "Các quy định về bảo hiểm hành lý là gì?",
    "Làm thế nào để tính lãi suất tiền vay?",
    "Pháp luật quy định như thế nào về quyền sở hữu nông nghiệp?",
    "Làm thế nào để đăng ký thi chứng chỉ CISSP?",
    "Các khoản phí khi mở tài khoản ví điện tử là gì?",
    "Làm thế nào để đăng ký bảo hiểm du học?",
    "Các quy định về bảo hiểm hàng không là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư cổ phiếu?",
    "Pháp luật quy định như thế nào về quyền tự do tín ngưỡng?",
    "Làm thế nào để đăng ký thi chứng chỉ PMP?",
    "Các khoản phí khi gửi tiền qua ngân hàng nước ngoài là gì?",
    "Làm thế nào để đăng ký học ngoại ngữ tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hải là gì?",
    "Làm thế nào để tính lãi suất tiền gửi tiết kiệm?",
    "Pháp luật quy định như thế nào về quyền sở hữu thương mại?",
    "Làm thế nào để đăng ký thi chứng chỉ CISM?",
    "Các khoản phí khi mở tài khoản chứng khoán quốc tế là gì?",
    "Làm thế nào để đăng ký bảo hiểm du lịch nước ngoài?",
    "Các quy định về bảo hiểm hàng hóa xuất nhập khẩu là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư trái phiếu chính phủ?",
    "Pháp luật quy định như thế nào về quyền tự do tôn giáo?",
    "Làm thế nào để đăng ký thi chứng chỉ PRINCE2?",
    "Các khoản phí khi gửi tiền qua ngân hàng liên ngân hàng là gì?",
    "Làm thế nào để đăng ký học kỹ năng mềm tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hóa nội địa là gì?",
    "Làm thế nào để tính lãi suất tiền gửi có kỳ hạn?",
    "Pháp luật quy định như thế nào về quyền sở hữu công nghệ?",
    "Làm thế nào để đăng ký thi chứng chỉ ITIL?",
    "Các khoản phí khi mở tài khoản chứng khoán nội địa là gì?",
    "Làm thế nào để đăng ký bảo hiểm du lịch trong nước?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường bộ là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư trái phiếu doanh nghiệp?",
    "Pháp luật quy định như thế nào về quyền tự do tổ chức hội?",
    "Làm thế nào để đăng ký thi chứng chỉ COBIT?",
    "Các khoản phí khi gửi tiền qua ngân hàng liên kết là gì?",
    "Làm thế nào để đăng ký học kỹ năng giao tiếp tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường thủy là gì?",
    "Làm thế nào để tính lãi suất tiền gửi không kỳ hạn?",
    "Pháp luật quy định như thế nào về quyền sở hữu nội thất?",
    "Làm thế nào để đăng ký thi chứng chỉ TOGAF?",
    "Các khoản phí khi mở tài khoản chứng khoán quốc tế là gì?",
    "Làm thế nào để đăng ký bảo hiểm du lịch quốc tế?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường hàng không là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư trái phiếu chính phủ nước ngoài?",
    "Pháp luật quy định như thế nào về quyền tự do tổ chức công đoàn?",
    "Làm thế nào để đăng ký thi chứng chỉ SCP?",
    "Các khoản phí khi gửi tiền qua ngân hàng liên doanh là gì?",
    "Làm thế nào để đăng ký học kỹ năng lãnh đạo tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường sắt là gì?",
    "Làm thế nào để tính lãi suất tiền gửi có kỳ hạn dài hạn?",
    "Pháp luật quy định như thế nào về quyền sở hữu nội thất nhà ở?",
    "Làm thế nào để đăng ký thi chứng chỉ PECB?",
    "Các khoản phí khi mở tài khoản chứng khoán nước ngoài là gì?",
    "Làm thế nào để đăng ký bảo hiểm du lịch quốc tế nhiều nước?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường hàng không quốc tế là gì?",
    "Làm thế nào để tính lợi nhuận đầu tư trái phiếu chính phủ nước ngoài dài hạn?",
    "Pháp luật quy định như thế nào về quyền tự do tổ chức đoàn thể?",
    "Làm thế nào để đăng ký thi chứng chỉ CGEIT?",
    "Các khoản phí khi gửi tiền qua ngân hàng liên doanh nước ngoài là gì?",
    "Làm thế nào để đăng ký học kỹ năng quản lý tại một trung tâm?",
    "Các quy định về bảo hiểm hàng hóa vận chuyển đường sắt nội địa là gì?",
    "Làm thế nào để tính lãi suất tiền gửi có kỳ hạn ngắn hạn?",
    "Pháp luật quy định như thế nào về quyền sở hữu nội thất công trình?",
    "Làm thế nào để đăng ký thi chứng chỉ CRISC?",
    "Quy định của Việt Nam về quyền tác giả đối với tác phẩm nghệ thuật là gì?",
    "Các bước để đăng ký bằng sáng chế tại Việt Nam như thế nào?",
    "Hợp đồng lao động cần phải bao gồm những nội dung gì theo luật định?",
    "Quy trình ly hôn tại Việt Nam diễn ra như thế nào?",
    "Người tiêu dùng có những quyền gì khi mua sản phẩm?",
    "Các hình thức xử phạt đối với hành vi vi phạm bản quyền là gì?",
    "Cần những điều kiện gì để thành lập một công ty luật?",
    "Quyền và nghĩa vụ của người thuê nhà là gì?",
    "Các bước để khiếu nại một quyết định hành chính là gì?",
    "Pháp luật Việt Nam quy định như thế nào về quyền sở hữu trí tuệ?",
    "Quy trình đăng ký nhập học tại các trường đại học ở Việt Nam là gì?",
    "Các loại học bổng có sẵn cho sinh viên Việt Nam là gì?",
    "Các yêu cầu về chương trình học cho bậc phổ thông là gì?",
    "Các tiêu chuẩn để trở thành giáo viên ở Việt Nam là gì?",
    "Các chính sách giáo dục hiện hành của政府 là gì?",
    "Các quy định về học online tại Việt Nam là gì?",
    "Các quyền của sinh viên trong trường đại học là gì?",
    "Các tiêu chuẩn đánh giá chất lượng trường học là gì?",
    "Các nguồn tài chính cho giáo dục ở Việt Nam là gì?",
    "Các hướng dẫn về phát triển chương trình giảng dạy là gì?",
    "Các dịch vụ ngân hàng cơ bản mà các ngân hàng tại Việt Nam cung cấp là gì?",
    "Các quy định về thuế thu nhập cá nhân ở Việt Nam là gì?",
    "Các lựa chọn đầu tư phổ biến ở Việt Nam là gì?",
    "Các bước để申请 một khoản vay tại ngân hàng là gì?",
    "Các loại bảo hiểm mà cá nhân có thể mua ở Việt Nam là gì?",
    "Các yêu cầu báo cáo tài chính theo luật định là gì?",
    "Các chính sách tiền tệ của Ngân hàng Nhà nước là gì?",
    "Các quy định về thuế doanh nghiệp ở Việt Nam là gì?",
    "Các luật về đầu tư chứng khoán ở Việt Nam là gì?",
    "Các hướng dẫn về lập kế hoạch tài chính cá nhân là gì?",
    "Pháp luật quy định như thế nào về quyền thừa kế tài sản?",
    "Làm thế nào để đăng ký bản quyền phần mềm tại Việt Nam?",
    "Các quy định về lao động ngoài giờ làm việc là gì?",
    "Quy định pháp luật về bảo vệ quyền lợi người tiêu dùng là gì?",
    "Các yêu cầu để đăng ký kinh doanh online tại Việt Nam là gì?",
    "Làm thế nào để xin giấy phép lao động cho người nước ngoài?",
    "Quy định về đăng ký thường trú tại Việt Nam là gì?",
    "Làm thế nào để khởi kiện tranh chấp đất đai?",
    "Pháp luật quy định như thế nào về quyền sử dụng đất trong hôn nhân?",
    "Làm thế nào để đăng ký quyền sở hữu công trình xây dựng?",
    "Quy trình đăng ký bản quyền nhạc và phim là gì?",
    "Các quy định về đăng ký sản phẩm xuất khẩu là gì?",
    "Pháp luật quy định như thế nào về bảo vệ dữ liệu cá nhân?",
    "Làm thế nào để khiếu nại vi phạm hợp đồng lao động?",
    "Các quy định pháp luật về bảo vệ môi trường là gì?",
    "Làm thế nào để xin giấy phép xây dựng nhà ở?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với sáng chế?",
    "Làm thế nào để đăng ký bằng sáng chế quốc tế từ Việt Nam?",
    "Các quy định về bảo hiểm thất nghiệp cho người lao động thời vụ là gì?",
    "Pháp luật quy định như thế nào về hợp đồng mua bán tài sản?",
    "Làm thế nào để xin giấy phép kinh doanh vận tải?",
    "Các bước để đăng ký quyền sở hữu nhãn hiệu tại Việt Nam là gì?",
    "Pháp luật quy định như thế nào về xử lý nợ xấu?",
    "Làm thế nào để xin giấy phép kinh doanh lữ hành quốc tế?",
    "Các quy định về hợp đồng thuê lao động thời vụ là gì?",
    "Pháp luật quy định như thế nào về quyền nuôi con sau ly hôn?",
    "Làm thế nào để giải quyết tranh chấp trong hợp đồng dân sự?",
    "Các quy định về bảo hiểm hưu trí bổ sung tại Việt Nam là gì?",
    "Pháp luật quy định như thế nào về bảo hộ chỉ dẫn địa lý?",
    "Làm thế nào để đăng ký quyền sở hữu cây giống?",
    "Quy trình khiếu kiện vi phạm bản quyền phần mềm là gì?",
    "Các quy định về thuế thu nhập doanh nghiệp đối với công ty khởi nghiệp là gì?",
    "Làm thế nào để xin cấp giấy chứng nhận an toàn thực phẩm?",
    "Các bước để đăng ký bảo hộ logo công ty là gì?",
    "Pháp luật quy định như thế nào về quyền của cổ đông trong công ty?",
    "Làm thế nào để đăng ký giấy phép hoạt động trong lĩnh vực tài chính?",
    "Các quy định về sử dụng lao động dưới 18 tuổi là gì?",
    "Pháp luật quy định như thế nào về xử lý rác thải công nghiệp?",
    "Làm thế nào để đăng ký bảo hiểm y tế tự nguyện?",
    "Các quy định về sử dụng vốn đầu tư công là gì?",
    "Pháp luật quy định như thế nào về hoạt động của các quỹ đầu tư?",
    "Làm thế nào để xin giấy phép khai thác khoáng sản?",
    "Các quy định về phân chia tài sản trong doanh nghiệp liên doanh là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với phần mềm mã nguồn mở?",
    "Làm thế nào để đăng ký sáng chế liên quan đến công nghệ xanh?",
    "Các quy định về hợp đồng bảo hiểm nhân thọ tại Việt Nam là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với giống cây trồng?",
    "Làm thế nào để xin cấp giấy phép hoạt động trong lĩnh vực giáo dục?",
    "Các quy định về quản lý nhà chung cư là gì?",
    "Pháp luật quy định như thế nào về đăng ký sáng chế trong lĩnh vực y tế?",
    "Làm thế nào để đăng ký quyền sở hữu phần mềm tại nước ngoài?",
    "Các quy định về trách nhiệm bảo vệ môi trường của doanh nghiệp là gì?",
    "Pháp luật quy định như thế nào về bảo vệ quyền lợi người lao động nữ?",
    "Làm thế nào để đăng ký bảo hộ quyền sở hữu trí tuệ đối với sản phẩm handmade?",
    "Các quy định về trách nhiệm xã hội của doanh nghiệp tại Việt Nam là gì?",
    "Pháp luật quy định như thế nào về quyền tự do kinh doanh?",
    "Làm thế nào để xin cấp giấy phép hoạt động trong lĩnh vực công nghệ?",
    "Các bước để đăng ký bản quyền sáng tạo nội dung số là gì?",
    "Pháp luật quy định như thế nào về quyền khởi kiện tranh chấp thương mại?",
    "Làm thế nào để đăng ký giấy phép nhập khẩu thiết bị y tế?",
    "Các quy định về quyền và nghĩa vụ của nhà đầu tư nước ngoài là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ thông tin cá nhân trực tuyến?",
    "Làm thế nào để đăng ký hợp đồng bảo hiểm sức khỏe?",
    "Các bước để khởi kiện vi phạm quyền sở hữu trí tuệ là gì?",
    "Pháp luật quy định như thế nào về quyền của người khuyết tật trong lao động?",
    "Làm thế nào để đăng ký bản quyền thiết kế thời trang?",
    "Các quy định về thuế đối với sản phẩm nhập khẩu là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ thương hiệu trong kinh doanh?",
    "Làm thế nào để xin giấy phép kinh doanh trong lĩnh vực dịch vụ ăn uống?",
    "Các quy định về đóng bảo hiểm xã hội tự nguyện là gì?",
    "Pháp luật quy định như thế nào về bảo vệ bí mật kinh doanh?",
    "Làm thế nào để đăng ký sở hữu trí tuệ đối với ứng dụng di động?",
    "Các bước để xin giấy phép hoạt động trong lĩnh vực năng lượng tái tạo là gì?",
    "Pháp luật quy định như thế nào về quyền của người lao động trong hợp đồng tập sự?",
    "Làm thế nào để đăng ký quyền sử dụng đất tại khu đô thị mới?",
    "Các quy định về xử lý vi phạm trong lĩnh vực quảng cáo là gì?",
    "Pháp luật quy định như thế nào về quyền bảo hộ tên thương mại?",
    "Làm thế nào để xin giấy phép tổ chức sự kiện quốc tế tại Việt Nam?",
    "Các bước để khởi kiện vi phạm hợp đồng mua bán nhà đất là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ người tiêu dùng đối với sản phẩm lỗi?",
    "Làm thế nào để xin cấp giấy phép hoạt động trong lĩnh vực truyền thông?",
    "Các quy định về sử dụng tài nguyên thiên nhiên là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với sách điện tử?",
    "Làm thế nào để đăng ký bảo hộ sáng chế trong lĩnh vực khoa học dữ liệu?",
    "Các bước để xử lý tranh chấp lao động tại tòa án là gì?",
    "Pháp luật quy định như thế nào về trách nhiệm xã hội của doanh nghiệp trong môi trường?",
    "Làm thế nào để xin giấy phép sản xuất thực phẩm chức năng?",
    "Các quy định về hợp đồng gia công sản phẩm là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ nhà đầu tư nhỏ lẻ?",
    "Làm thế nào để đăng ký quyền sở hữu trí tuệ đối với nhãn hiệu quốc tế?",
    "Các quy định về đóng thuế tài nguyên là gì?",
    "Pháp luật quy định như thế nào về quyền bảo hộ giống động vật mới?",
    "Làm thế nào để đăng ký giấy phép sản xuất phân bón hữu cơ?",
    "Các bước để xử lý tranh chấp hợp đồng thương mại quốc tế là gì?",
    "Pháp luật quy định như thế nào về quyền tác giả đối với ảnh chụp?",
    "Làm thế nào để xin giấy phép nhập khẩu các sản phẩm nông nghiệp?",
    "Các quy định về bảo hiểm hưu trí doanh nghiệp là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với sản phẩm nghệ thuật số?",
    "Làm thế nào để đăng ký bảo hộ thương hiệu tại ASEAN?",
    "Các quy định về xử lý vi phạm trong lĩnh vực môi trường là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ dữ liệu người tiêu dùng?",
    "Làm thế nào để xin giấy phép hoạt động giáo dục tư thục?",
    "Các bước để đăng ký quyền sở hữu trí tuệ đối với tác phẩm âm nhạc là gì?",
    "Pháp luật quy định như thế nào về bảo vệ lợi ích công cộng trong sử dụng đất đai?",
    "Làm thế nào để đăng ký sáng chế trong lĩnh vực trí tuệ nhân tạo?",
    "Các quy định về trách nhiệm bảo vệ môi trường trong sản xuất công nghiệp là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với công nghệ blockchain?",
    "Làm thế nào để xin cấp giấy phép kinh doanh vận tải hành khách quốc tế?",
    "Các bước để đăng ký hợp đồng bảo hiểm phi nhân thọ là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ thương mại điện tử?",
    "Làm thế nào để đăng ký giấy phép khai thác du lịch sinh thái?",
    "Các quy định về xử lý vi phạm bản quyền thiết kế đồ họa là gì?",
    "Pháp luật quy định như thế nào về quyền sở hữu trí tuệ đối với phần mềm AI?",
    "Làm thế nào để đăng ký bảo hộ sáng chế trong lĩnh vực y học?",
    "Các bước để xin giấy phép hoạt động trong lĩnh vực y tế là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ thông tin doanh nghiệp?",
    "Làm thế nào để đăng ký bảo hiểm tai nạn lao động cho doanh nghiệp?",
    "Các quy định về đóng thuế thu nhập cá nhân đối với người nước ngoài là gì?",
    "Pháp luật quy định như thế nào về quyền bảo vệ thông tin cá nhân trên mạng xã hội?",
    "Làm thế nào để xin giấy phép kinh doanh xuất nhập khẩu?",
    "Các quy định về sử dụng lao động trong ngành công nghiệp nặng là gì?"
]

def download_parquet_files(output_dir, num_files=132):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset/resolve/main/data"
    
    for i in range(num_files):
        file_name = f"train-{i:05d}-of-00132.parquet"
        output_file = os.path.join(output_dir, file_name)
        
        if os.path.exists(output_file):
            continue
        
        try:
            url = f"{base_url}/{file_name}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        except Exception as e:
            logging.error(f"Error downloading file {i}: {str(e)}")

def get_checkpoint_path(output_file):
    return f"{output_file}.checkpoint"

def save_checkpoint(checkpoint_path, processed_count):
    with open(checkpoint_path, 'w') as f:
        json.dump({"processed_count": processed_count}, f)
    logging.info(f"Checkpoint saved: processed_count={processed_count}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data["processed_count"]
    return 0

def save_results(output_file, results):
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    logging.info(f"Appended {len(results)} results to {output_file}")

def process_parquet_file(input_file, output_file, queries_list, model_name="bkai-foundation-models/vietnamese-bi-encoder", batch_size=32):
    checkpoint_path = get_checkpoint_path(output_file)
    processed_count = load_checkpoint(checkpoint_path)
    logging.info(f"Loaded checkpoint: processed_count={processed_count}")
    import torch
    model = SentenceTransformer(model_name,model_kwargs={"torch_dtype": torch.float16})
    instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
    prompt = f'<instruct>{instruction}\n<query>'
    df = pd.read_parquet(input_file)
    texts = df['text'].tolist()
    
    passages = []
    for i in range(1, len(texts), 3):
        if i + 1 < len(texts):
            passages.append((texts[i], texts[i+1]))
    
    passages = passages[processed_count:]
    
    for i in tqdm(range(0, len(passages), batch_size)):
        batch_passages = passages[i:i+batch_size]
        
        batch_queries = random.choices(queries_list, k=len(batch_passages))
        
        query_embeddings = model.encode(batch_queries,prompt=prompt)
        
        passages2 = [pair[0] for pair in batch_passages]
        passages3 = [pair[1] for pair in batch_passages]
        emb2 = model.encode(passages2)
        emb3 = model.encode(passages3)
        
        sim_q2 = model.similarity_pairwise(query_embeddings, emb2)
        sim_q3 = model.similarity_pairwise(query_embeddings, emb3)
        
        batch_results = []
        for q, p2, p3, s_q2, s_q3 in zip(batch_queries, passages2, passages3, sim_q2, sim_q3):
            result = {
                "query": q,
                "passage2": p2,
                "passage3": p3,
                "label": float(s_q2 - s_q3)
            }
            batch_results.append(result)
        
        save_results(output_file, batch_results)
        
        processed_count += len(batch_passages)
        save_checkpoint(checkpoint_path, processed_count)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    logging.info(f"Processing completed for {input_file}")

def process_directory(input_dir, output_dir, queries_list, model_name="bkai-foundation-models/vietnamese-bi-encoder"):
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    
    for file in parquet_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
        try:
            process_parquet_file(input_path, output_path, queries_list, model_name)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    input_directory = "vietnamese_dataset"
    download_parquet_files(input_directory, num_files=20)
    output_directory = "processed_dataset"
    model_name = "BAAI/bge-multilingual-gemma2"
    process_directory(input_directory, output_directory, queries, model_name)