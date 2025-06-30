import numpy as np
import pdb
import cv2
import os
import json


def getFileList(dir, Filelist, ext=None, skip=None, spec=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, s)):
                newDir = os.path.join(dir, s)
                getFileList(newDir, Filelist, ext, skip, spec)

            else:
                acpt = True
                if skip is not None:
                    for skipi in skip:
                        if skipi in s:
                            acpt = False
                            break

                if acpt == False:
                    continue
                else:
                    sp = False
                    if spec is not None:
                        for speci in spec:
                            if speci in s:
                                sp = True
                                break

                    else:
                        sp = True

                    if sp == False:
                        continue
                    else:
                        newDir = os.path.join(dir, s)
                        getFileList(newDir, Filelist, ext, skip, spec)

    return Filelist


def transfer2(image, mapping):
    image_base = image.copy()

    for i in mapping.keys():
        image[image_base == i] = mapping[i]

    return image


image_dir = "/path-to-gt/"
transfer_image_written_dir = "/path-to-mapping-gt/"

mapping = {3:2, 4:2, 5:3, 6:4, 7:5, 8:6, 9:6, 10:7, 11:7, 12:8, 13:9, 14:10, 15:11, 16:12, 17:13, 18:14, 19:15}


imglist = getFileList(image_dir, [])
print(len(imglist))
x = 0
for img_d in imglist:
    print(x)
    img_name = img_d.split('/')[-1]
    sample = img_d.split('/')[-2]
    image = cv2.imread(img_d, 0)

    image_t = transfer2(image, mapping)
    os.makedirs(os.path.join(transfer_image_written_dir, sample), exist_ok=True)

    cv2.imwrite(os.path.join(transfer_image_written_dir, sample, img_name), image_t)
    x+=1

