import os

# Danh sách các thư mục con trực tiếp cần đọc
folders = [ "src"]

# Tên file xuất kết quả
output_filename = "content1.txt"

with open(output_filename, "w", encoding="utf-8") as outfile:
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Bỏ qua: Không tìm thấy thư mục '{folder}'")
            continue

        # os.walk duyệt qua tất cả thư mục và file ở mọi cấp con
        for root, dirs, files in os.walk(folder):
            for file in files:
                # Chỉ xử lý các file có đuôi .py
                if not file.endswith(".py"):
                    continue

                file_path = os.path.join(root, file)

                # Chuẩn hóa đường dẫn dùng dấu '/' thay vì '\' trên Windows
                normalized_path = file_path.replace("\\", "/")

                # Bỏ qua nếu chính là file kết quả (nếu file kết quả vô tình đuôi .py)
                if os.path.abspath(file_path) == os.path.abspath(output_filename):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        content = infile.read()

                    # Ghi định dạng: ====<file path>\n<nội dung file>\n
                    outfile.write(f"===={normalized_path}\n{content}\n")

                except UnicodeDecodeError:
                    print(f"Bỏ qua file không đúng mã UTF-8: {normalized_path}")
                except Exception as e:
                    print(f"Lỗi khi đọc file {normalized_path}: {e}")

print(f"Hoàn thành! Đã ghi nội dung các file .py vào {output_filename}")