import os
import shutil
import random


if __name__ == '__main__':
    val_root_dir = './data/val'
    train_root_dir = './data/train/'
    if not os.path.exists(val_root_dir):
        os.makedirs(val_root_dir)

    for root, _, imgs in os.walk(train_root_dir):
        random.shuffle(imgs)
        split_index = int(len(imgs) * 0.2)
        dst_root = os.path.join(val_root_dir, root.split('/')[-1])
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        for img in imgs[:split_index]:
            src = os.path.join(root, img)
            dst = os.path.join(dst_root, img)
            shutil.move(src, dst)
