import os
import sys
import re

def extract_gdrive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def download_datasets(list_file="dataset.txt", output_dir="./data"):
    if not os.path.exists(list_file):
        print(f"[!] Không tìm thấy file: {list_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cookie_file = os.path.join(output_dir, ".sharepoint_cookie.txt")
    
    # Giả lập header trình duyệt thật để tránh lỗi HTTP 429
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    with open(list_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    print(f"[*] Bắt đầu tải {len(lines)} dataset vào '{output_dir}'...")

    for i, line in enumerate(lines, 1):
        parts = line.split()
        raw_url = parts[0]
        out_name = parts[1] if len(parts) > 1 else None

        print(f"\n[{i}/{len(lines)}] Đang xử lý: {raw_url}")

        # 1. Xử lý Google Drive qua endpoint direct
        if "drive.google.com" in raw_url:
            file_id = extract_gdrive_id(raw_url)
            if not file_id:
                print(f"[!] Không tìm thấy ID từ URL: {raw_url}")
                continue

            target_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=yes"
            filename = out_name if out_name else f"{file_id}.zip"
            out_path = os.path.join(output_dir, filename)

            cmd = (
                f'wget -c --no-check-certificate '
                f'--user-agent="{user_agent}" '
                f'-O "{out_path}" "{target_url}"'
            )
            os.system(cmd)

        # 2. Xử lý SharePoint / OneDrive (kèm cookie và User-Agent)
        elif "sharepoint.com" in raw_url or "1drv.ms" in raw_url:
            target_url = raw_url
            if "download=1" not in target_url:
                sep = "&" if "?" in target_url else "?"
                target_url += f"{sep}download=1"

            filename = out_name if out_name else "dataset_sharepoint.zip"
            out_path = os.path.join(output_dir, filename)

            if os.path.exists(cookie_file):
                os.remove(cookie_file)

            # Thêm --save-cookies, --load-cookies và --keep-session-cookies để giữ session khi 302 redirect
            cmd = (
                f'wget -c --no-check-certificate '
                f'--user-agent="{user_agent}" '
                f'--save-cookies "{cookie_file}" --load-cookies "{cookie_file}" --keep-session-cookies '
                f'-O "{out_path}" "{target_url}"'
            )
            exit_code = os.system(cmd)

            # Dự phòng sang curl nếu wget vẫn bị 429 rate-limit
            if exit_code != 0:
                print("[*] Thử lại bằng curl...")
                curl_cmd = (
                    f'curl -L -k -C - -A "{user_agent}" '
                    f'-b "{cookie_file}" -c "{cookie_file}" '
                    f'-o "{out_path}" "{target_url}"'
                )
                os.system(curl_cmd)

            if os.path.exists(cookie_file):
                os.remove(cookie_file)

        # 3. Link HTTP/HTTPS thông thường
        else:
            if out_name:
                out_path = os.path.join(output_dir, out_name)
                cmd = f'wget -c --no-check-certificate --user-agent="{user_agent}" -O "{out_path}" "{raw_url}"'
            else:
                cmd = f'wget -c --no-check-certificate --user-agent="{user_agent}" -P "{output_dir}" "{raw_url}"'
            os.system(cmd)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.txt"
    download_datasets(file_path)