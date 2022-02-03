import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd()
classes = ["no_mask","face","face_mask","nomask","face_nask"]

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id),encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    global no_mask_count, mask_count
    if root.find('object')==None:
        return
    # list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    list_file.write('VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id))
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        if class_name == "no_mask" or class_name =="face" or class_name =="nomask":
            cls_id = 0
            no_mask_count += 1
        else:
            cls_id = 1
            mask_count += 1
        print(class_name, cls_id)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # list_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    list_file.write('\n')

no_mask_count = 0
mask_count = 0

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set),encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w',encoding='utf-8')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()

print('Num of no_mask:', no_mask_count)
print('Num of mask:', mask_count)
print ('Note: some images may have multiply no_masks or masks, so the sum of them are larger than the image number.')