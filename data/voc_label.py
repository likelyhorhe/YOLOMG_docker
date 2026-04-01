import os
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sets = ['train', 'val', 'test']
base = os.path.join(PROJECT_ROOT, 'datasets', 'ARD100_mask32')

classes = ['Drone']


def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(image_id):
    xml_path = '%s/Annotations/%s.xml' % (base, image_id)
    label_path = '%s/labels/%s.txt' % (base, image_id)

    if not os.path.exists(xml_path):
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(label_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            out_file.write('%d %.6f %.6f %.6f %.6f\n' % (cls_id, bb[0], bb[1], bb[2], bb[3]))


# Step 1: Convert all XML annotations to YOLO format
print('Converting XML annotations to YOLO format...')
label_dir = '%s/labels' % base
os.makedirs(label_dir, exist_ok=True)

anno_dir = '%s/Annotations' % base
converted = 0
for xml_file in os.listdir(anno_dir):
    if xml_file.endswith('.xml'):
        image_id = xml_file[:-4]
        convert_annotation(image_id)
        converted += 1
print('Converted %d annotations' % converted)

# Step 2: Generate image list files
for image_set in sets:
    image_ids = open('%s/ImageSets/Main/%s.txt' % (base, image_set)).read().strip().split()

    list_file = open('%s/%s.txt' % (base, image_set), 'w')
    list_file2 = open('%s/%s2.txt' % (base, image_set), 'w')

    for image_id in image_ids:
        list_file.write('%s/images/%s.jpg\n' % (base, image_id))
        list_file2.write('%s/masks/%s.jpg\n' % (base, image_id))

    list_file.close()
    list_file2.close()

print('Done.')
