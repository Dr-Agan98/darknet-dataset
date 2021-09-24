import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import glob

ia.seed(1)

img_paths_p = glob.glob("/mnt/e/Users/Draga/Documents/darknet/orov_custom/base_images/*")
bb_paths_p = glob.glob("/mnt/e/Users/Draga/Documents/darknet/orov_custom/base_labels/*")
img_paths = img_paths_p + img_paths_p + img_paths_p + img_paths_p + img_paths_p
bb_paths = bb_paths_p + bb_paths_p + bb_paths_p + bb_paths_p + bb_paths_p
images=[]
bbs=[]

for idx, (img_path,bb_path) in enumerate(zip(img_paths,bb_paths)):
    img = cv2.imread(img_path, 1)
    img_w = img.shape[1]
    img_h = img.shape[0]
    bb_file = open(bb_path, "r")
    bbs_coords = bb_file.readlines()
    img_bbs = []
    for bb_coords in bbs_coords:
        bb_coord = bb_coords.split()
        x_c = float(bb_coord[1])*img_w
        y_c = float(bb_coord[2])*img_h
        w = float(bb_coord[3])*img_w
        h = float(bb_coord[4])*img_h

        x1=x_c-(w/2)
        y1=y_c-(h/2)
        x2=x_c+(w/2)
        y2=y_c+(h/2)

        img_bbs.append( BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) )
        
    images.append(img)
    bbs.append( BoundingBoxesOnImage(img_bbs, shape=img.shape) )

#bbs = BoundingBoxesOnImage([
#    BoundingBox(x1=65, y1=100, x2=200, y2=150),
#    BoundingBox(x1=150, y1=80, x2=200, y2=130)
#], shape=image.shape)

seq = iaa.Sequential([
    iaa.Fliplr(0.4),
    iaa.Flipud(0.6),
    iaa.Sometimes( 0.7,iaa.GaussianBlur(sigma=(1, 2.5)) ),
    iaa.Sometimes( 0.3, iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255)) ),
    iaa.Sometimes( 0.8, iaa.Cutout(nb_iterations=(2, 4), size=0.25, squared=False) ),
    iaa.Multiply((0.6, 1.2)),
    iaa.Sometimes(0.9, iaa.Affine(
        translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        scale=(0.2,0.7),
        rotate=(-45, 45)
    ))
])

# Augment BBs and images.
images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)

for idx, (image_aug, bb_aug) in enumerate(zip(images_aug, bbs_aug)):
    img_w = image_aug.shape[1]
    img_h = image_aug.shape[0]
    label = open("/mnt/e/Users/Draga/Documents/darknet/orov_custom/labels-aug/img-aug-"+str(idx)+".txt", "w")
    for bb in bb_aug.bounding_boxes:
        bb_x1 = bb.x1
        bb_y1 = bb.y1
        bb_x2 = bb.x2
        bb_y2 = bb.y2

        bb_w = (bb_x2-bb_x1)
        bb_h = (bb_y2-bb_y1)
        bb_x = (bb_x1+(bb_w/2))
        bb_y = (bb_y1+(bb_h/2))
        
        label.write("0 "+str(bb_x/img_w)+" "+str(bb_y/img_h)+" "+str(bb_w/img_w)+" "+str(bb_h/img_h)+"\n")
        
    label.close()    
    cv2.imwrite("/mnt/e/Users/Draga/Documents/darknet/orov_custom/images-aug/img-aug-"+str(idx)+".jpg",image_aug)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
'''for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )'''

# image with BBs before/after augmentation (shown below)
#image_before = bbs.draw_on_image(image, size=2)
#image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
