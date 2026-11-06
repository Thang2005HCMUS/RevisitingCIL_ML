import os
import sys
import subprocess

def download_models(list_file="models.txt", output_dir="./pretrained_models"):
    if not os.path.exists(list_file):
        print(f"[!] Không tìm thấy file: {list_file}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(list_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"[*] Bắt đầu tải {len(urls)} model vào thư mục '{output_dir}'...")

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Đang tải: {url}")
        
        # -c: tiếp tục tải nếu đứt đoạn
        # --no-check-certificate: bỏ qua lỗi SSL
        # -P: lưu vào thư mục chỉ định
        cmd = [
            "wget",
            "-c",
            "--no-check-certificate",
            "-P", output_dir,
            url
        ]

        try:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"[!] Lỗi khi tải: {url} (Exit code: {res.returncode})")
            else:
                print(f"[+] Hoàn tất: {url}")
        except Exception as e:
            print(f"[!] Ngoại lệ: {e}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "models.txt"
    download_models(file_path)