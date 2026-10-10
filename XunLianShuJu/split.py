import os
import random
import shutil

root = r"F:\WaiBaoDaSai\XunLianShuJu"
img_src = os.path.join(root, "images")
label_src = os.path.join(root, "labels")

train_img = os.path.join(root, "images_split", "train")
val_img = os.path.join(root, "images_split", "val")
train_label = os.path.join(root, "labels_split", "train")
val_label = os.path.join(root, "labels_split", "val")

random.seed(42)

img_list = [f for f in os.listdir(img_src) if f.lower().endswith(".jpg")]
random.shuffle(img_list)

split_point = int(len(img_list)*0.8)
train_list = img_list[:split_point]
val_list = img_list[split_point:]

print(f"总图片数量：{len(img_list)}")
print(f"训练集图片：{len(train_list)}")
print(f"验证集图片：{len(val_list)}")

def copy_data(img_name, target_img_folder, target_label_folder):
    try:
        src_img = os.path.join(img_src, img_name)
        shutil.copy(src_img, target_img_folder)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(label_src, txt_name)
        if os.path.exists(src_label):
            shutil.copy(src_label, target_label_folder)
    except Exception as e:
        print(f"复制失败：{img_name}，错误：{e}")

# 复制训练集
print("开始复制训练集...")
for name in train_list:
    copy_data(name, train_img, train_label)
print("训练集复制完成！")

# 复制验证集
print("开始复制验证集...")
for name in val_list:
    copy_data(name, val_img, val_label)
print("全部拆分完成！")
