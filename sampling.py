import os
import shutil
import random

data_root = r"C:/Users/shuza/Desktop/Data Cleaned" 
output_root = r"C:/Users/shuza/Desktop/Data Final/Classification Data"

MIN_IMAGES = 250 
MAX_IMAGES = 2000

random.seed(42)
os.makedirs(output_root, exist_ok=True)

dropped = []
capped = []
kept = []

for class_folder in sorted(os.listdir(data_root)):
    class_path = os.path.join(data_root, class_folder)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.gif'))]
    count = len(images)

    if count < MIN_IMAGES:
        dropped.append((class_folder, count))
        continue

    if count > MAX_IMAGES:
        images = random.sample(images, MAX_IMAGES)
        capped.append((class_folder, count))

    out_path = os.path.join(output_root, class_folder)
    os.makedirs(out_path, exist_ok=True)
    for img in images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(out_path, img)
        )
    kept.append((class_folder, min(count, MAX_IMAGES)))

print(f"Kept:    {len(kept)} classes")
print(f"Capped:  {len(capped)} classes")
print(f"Dropped: {len(dropped)} classes")
print(f"Dropped classes:")
for name, count in sorted(dropped, key=lambda x: x[1]):
    print(f"  {count:>5}  {name}")


