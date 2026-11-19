import json
import os

PATCH_FILE = "patch.json"

with open(PATCH_FILE, "r", encoding="utf-8") as f:
    patches = json.load(f)

for rel_path, updates in patches.items():
    if not os.path.exists(rel_path):
        print(f"⚠️ Không tìm thấy file: {rel_path}")
        continue

    with open(rel_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Cập nhật giá trị mới cho các key tương ứng
    changed = []
    for k, new_val in updates.items():
        old_val = config_data.get(k, "<không có>")
        if old_val != new_val:
            config_data[k] = new_val
            changed.append(f"{k}: {old_val} -> {new_val}")

    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    if changed:
        print(f"✅ Đã sửa {rel_path} | {', '.join(changed)}")
    else:
        print(f"➖ Không có thay đổi: {rel_path}")