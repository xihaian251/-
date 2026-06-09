import os
import shutil
import glob

# 你的项目根目录和模型文件信息
project_dir = r"D:\python-study"
target_filename = "final_model.pth"  # 关键：根据你的实际文件名修改

# 1. 首先检查项目根目录下是否已经有了模型文件
# 注意：因为文件名可能不完全匹配，我们用更灵活的方式检查
found_in_project = False
project_possible_files = os.listdir(project_dir)
for file in project_possible_files:
    if file.lower().endswith(".pth"):  # 查找所有 .pth 结尾的文件
        found_in_project = True
        target_filename = file  # 更新变量为实际找到的文件名
        print(f"✅ 在项目目录中找到模型文件: {target_filename}")
        break

if not found_in_project:
    print("🔍 正在搜索模型文件...")
    # 2. 如果项目目录里没有，就在整个用户目录下搜索 .pth 文件
    search_root = os.path.expanduser("~")  # 相当于 C:\Users\<你的用户名>

    # 使用 glob 模块递归搜索所有 .pth 文件，这种方式更清晰
    # 为了避免搜索过慢，我们可以限定在 AppData 目录，因为 PyCharm 的设置都在那里
    search_path = os.path.join(search_root, "AppData", "Local", "JetBrains", "**", "*.pth")
    found_files = glob.glob(search_path, recursive=True)

    # 如果上面没找到，再在用户目录下进行全盘搜索
    if not found_files:
        print("在 AppData 目录下未找到，正在全盘搜索（可能需要一点时间）...")
        found_files = glob.glob(os.path.join(search_root, "**", "*.pth"), recursive=True)

    if found_files:
        # 假设我们找到了多个 .pth 文件，提示用户选择
        if len(found_files) > 1:
            print("🔎 找到多个可能的 .pth 文件：")
            for i, file_path in enumerate(found_files):
                print(f"{i + 1}: {file_path}")
            # 我们默认选择第一个，但更合理的是让用户决定
            chosen_path = found_files[0]
            print(f"✅ 将复制第一个找到的文件: {chosen_path}")
        else:
            chosen_path = found_files[0]
            print(f"✅ 找到文件: {chosen_path}")

        # 复制文件到项目目录
        destination_path = os.path.join(project_dir, target_filename)
        shutil.copy2(chosen_path, destination_path)  # copy2 会保留元数据
        print(f"✅ 文件已成功复制到: {destination_path}")
    else:
        print("❌ 未在任何位置找到 .pth 文件，请检查文件名或手动搜索。")