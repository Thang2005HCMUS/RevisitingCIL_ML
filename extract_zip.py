import os
import sys
import zipfile

def extract_archives(config_file="extract_zip.txt"):
    if not os.path.exists(config_file):
        print(f"[!] Không tìm thấy file cấu hình: {config_file}")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        print("[!] Không có tác vụ giải nén nào trong file cấu hình.")
        return

    print(f"[*] Bắt đầu giải nén {len(lines)} file...")

    for idx, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) < 2:
            print(f"[!] Dòng không hợp lệ ({line}). Cần: <đường dẫn zip> <thư mục đích>")
            continue

        zip_path, target_dir = parts[0], parts[1]

        if not os.path.exists(zip_path):
            print(f"\n[{idx}/{len(lines)}] [!] Bỏ qua: Không tìm thấy file {zip_path}")
            continue

        os.makedirs(target_dir, exist_ok=True)
        print(f"\n[{idx}/{len(lines)}] Đang giải nén '{zip_path}' vào '{target_dir}'...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"[+] Hoàn thành giải nén: {zip_path} -> {target_dir}")
        except zipfile.BadZipFile:
            print(f"[!] Lỗi: File zip bị hỏng hoặc không đúng định dạng ({zip_path})")
        except Exception as e:
            print(f"[!] Lỗi khi giải nén {zip_path}: {e}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "extract_zip.txt"
    extract_archives(cfg)