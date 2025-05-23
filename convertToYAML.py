import os
import shutil
from sklearn.model_selection import train_test_split

image_dir = "images-yolo"
label_dir = "labels-yolo"

output_image_dir = "images-spots"
output_label_dir = "labels-spots"

train_ratio = 0.8

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_basenames = [os.path.splitext(f)[0] for f in image_files]

train_files, val_files = train_test_split(image_basenames, train_size=train_ratio, random_state=42)

def copy_files(file_list, target_split):
    os.makedirs(f"{output_image_dir}/{target_split}", exist_ok=True)
    os.makedirs(f"{output_label_dir}/{target_split}", exist_ok=True)

    for base in file_list:
        img_src = None
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(image_dir, base + ext)
            if os.path.exists(path):
                img_src = path
                break

        lbl_src = os.path.join(label_dir, base + ".txt")

        if img_src: shutil.copy(img_src, os.path.join(output_image_dir, target_split, os.path.basename(img_src)))
        else: print(f"Image file not found for base: {base}")

        if os.path.exists(lbl_src): shutil.copy(lbl_src, os.path.join(output_label_dir, target_split, os.path.basename(lbl_src)))
        else:print(f"Label file not found for: {base}")

copy_files(train_files, "train")
copy_files(val_files, "val")

print("Dataset split complete.")
