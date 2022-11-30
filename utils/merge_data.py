import os
import shutil
import random


if __name__ == '__main__':
    val_root_dir = './data/val'
    train_root_dir = './data/train/'

    for root, _, imgs in os.walk(val_root_dir):
        dst_root = os.path.join(train_root_dir, root.split("/")[-1])
        for img in imgs:
            src = os.path.join(root, img)
            dst = os.path.join(dst_root, img)
            shutil.move(src, dst)
