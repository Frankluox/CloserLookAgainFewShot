# Adapted from CNAPS: https://github.com/cambridge-mlg/cnaps/blob/master/src/prepare_extra_datasets.py
"""
Process several raw datasets into well-processed, raw images.
"""
import gzip
import numpy as np
import os
import argparse
from PIL import Image
import pickle
import json
import itertools
from absl import logging
import collections


def process_mnist(datasrc_path, data_dst_path):
    """extract images from files"""
    def get_images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 28, 28)

    def get_labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        return integer_labels

    def create_image_dir(images, labels, path):
        if not os.path.exists(path):
            os.makedirs(path)
        class_counter = {}
        for cls in range(10):
            class_counter[cls] = 0
        for step, (image, label) in enumerate(zip(images, labels)):
            im = Image.fromarray(image)
            im = im.convert('RGB')
            im = im.resize((84, 84), resample=Image.LANCZOS)

            class_dir = os.path.join(path, 'class_%s' % label.item())
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            im_file = class_dir + '/image_%s.png' % (class_counter[label])
            im.save(im_file)
            class_counter[label] += 1
            if (step + 1) % 1000 == 0:
                print('Processed %s images.' % (step + 1))

    test_images = get_images(os.path.join(datasrc_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = get_labels(os.path.join(datasrc_path, 't10k-labels-idx1-ubyte.gz'))
    create_image_dir(test_images, test_labels, os.path.join(data_dst_path, 'mnist'))

def process_cifar(src_path, dst_path, imagesFileName, labelsFileName, labelKey1, labelKey2):
    """extract images from files"""
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def save_images_in_data_batch(source_dir, data_batch, data_dir, labelKey1, labelKey2):
        data_dict = unpickle(os.path.join(source_dir, data_batch))
        num_images = len(data_dict[labelKey1])
        for i in range(num_images):
            label = data_dict[labelKey1][i]
            label_name = (labels_dict[labelKey2][label]).decode('utf-8')
            class_dir = os.path.join(data_dir, label_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            file_name = (data_dict[b'filenames'][i]).decode('utf-8')
            image = data_dict[b'data'][i]
            image = np.reshape(image, (3, 32, 32))
            image = np.transpose(image, (1, 2, 0))
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((84, 84), resample=Image.LANCZOS)
            pil_image.save(os.path.join(class_dir, file_name))

    # load the label names
    labels_dict = unpickle(os.path.join(src_path, labelsFileName))

    # process test batch
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    save_images_in_data_batch(src_path, imagesFileName, dst_path, labelKey1, labelKey2)

def process_quickdraw(src_path, dst_path):
    """extract images from npy files"""
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    def load_image(img):
        """Load image img.
        Args:
        img: a 1D numpy array of shape [side**2]
        Returns:
        a PIL Image
        """
        # We make the assumption that the images are square.
        side = int(np.sqrt(img.shape[0]))
        # To load an array as a PIL.Image we must first reshape it to 2D.
        img = Image.fromarray(img.reshape((side, side)))
        return img
    for class_ in os.listdir(src_path):
        if "npy" not in class_:
            continue
        file_name = os.path.join(src_path, class_)
        with open(file_name, 'rb') as f:
            imgs = np.load(f)
        if not os.path.exists(f"{dst_path}/{class_[:-4]}"):
            os.makedirs(f"{dst_path}/{class_[:-4]}")
        for i, image in enumerate(imgs):
            img = load_image(image)
            img.save(f"{dst_path}/{class_[:-4]}/file_{i+1}.png")
            
def process_coco(src_path, 
                 dst_path, 
                 image_subdir_name='train2017',
                 annotation_json_name='annotations/instances_train2017.json',
                 box_scale_ratio=1.2):
    """extract objects from images"""
    num_all_classes = 80
    annotation_path = os.path.join(src_path, annotation_json_name)
    if not os.path.exists(annotation_path):
      raise ValueError('Annotation file %s does not exist' % annotation_path)
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    os.system(f"cp {annotation_path} {os.path.join(dst_path, annotation_json_name)}")

    with open(annotation_path, 'r') as json_file:
      annotations = json.load(json_file)
      instance_annotations = annotations['annotations']
      if not instance_annotations:
        raise ValueError('Instance annotations is empty.')
      coco_instance_annotations = instance_annotations
      categories = annotations['categories']
      if len(categories) != num_all_classes:
        raise ValueError(
            'Total number of MSCOCO classes %d should be equal to the sum of '
            'train, val, test classes %d.' %
            (len(categories), num_all_classes))
      coco_categories = categories

    coco_name_to_category = {cat['name']: cat for cat in categories}
    all_classes = list(coco_name_to_category.keys())
    if box_scale_ratio < 1.0:
        raise ValueError('Box scale ratio must be greater or equal to 1.0.')

    coco_id_to_classname = {}

    for class_id, class_name in enumerate(all_classes):
      category = coco_name_to_category[class_name]
      coco_id_to_classname[category['id']] = class_name

      if not os.path.exists(os.path.join(dst_path, class_name)):
            os.makedirs(os.path.join(dst_path, class_name))
        
    
    image_dir = os.path.join(src_path, image_subdir_name)

    def get_image_crop_and_class_id(annotation):
      """Gets image crop and its class label."""
      image_id = annotation['image_id']
      image_path = os.path.join(image_dir, '%012d.jpg' % image_id)
      # The bounding box is represented as (x_topleft, y_topleft, width, height)
      bbox = annotation['bbox']
      coco_class_id = annotation['category_id']
      classname = coco_id_to_classname[coco_class_id]

      with open(image_path, 'rb') as f:
        # The image shape is [?, ?, 3] and the type is uint8.
        image = Image.open(f)
        image = image.convert(mode='RGB')
        image_w, image_h = image.size

        def scale_box(bbox, scale_ratio):
          x, y, w, h = bbox
          x = x - 0.5 * w * (scale_ratio - 1.0)
          y = y - 0.5 * h * (scale_ratio - 1.0)
          w = w * scale_ratio
          h = h * scale_ratio
          return [x, y, w, h]

        x, y, w, h = scale_box(bbox, box_scale_ratio)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size
        if crop_width <= 0 or crop_height <= 0:
          raise ValueError('crops are not valid.')
      return image_crop, classname

    class_name_to_fileid = collections.defaultdict(int)

    for annotation in coco_instance_annotations:
        try:
            image_crop, classname = get_image_crop_and_class_id(annotation)
        except IOError:
            logging.warning('Image can not be opened and will be skipped.')
            continue
        except ValueError:
            logging.warning('Image can not be cropped and will be skipped.')
            continue    
        
        fileid = class_name_to_fileid[classname]+1
        file_name = 'file_'+'0'*(6-len(str(fileid)))+str(fileid)+'.png'
        image_crop.save(os.path.join(dst_path, classname, file_name))
        class_name_to_fileid[classname] += 1
     
def process_aircraft(src_path, 
                 dst_path):
    """cut images"""
    variants_path = os.path.join(src_path, 'data', 'variants.txt')
    with open(variants_path, 'r') as f:
      all_classes = [line.strip() for line in f.readlines() if line]
    assert len(all_classes)==100

    # Retrieve mapping from filename to bounding box.
    # Cropping to the bounding boxes is important for two reasons:
    # 1) The dataset documentation mentions that "[the] (main) aircraft in each
    #    image is annotated with a tight bounding box [...]", which suggests
    #    that there may be more than one aircraft in some images. Cropping to
    #    the bounding boxes removes ambiguity as to which airplane the label
    #    refers to.
    # 2) Raw images have a 20-pixel border at the bottom with copyright
    #    information which needs to be removed. Cropping to the bounding boxes
    #    has the side-effect that it removes the border.

    bboxes_path = os.path.join(src_path, 'data', 'images_box.txt')
    with open(bboxes_path, 'r') as f:
      names_to_bboxes = [
          line.split('\n')[0].split(' ') for line in f.readlines()
      ]
      names_to_bboxes = dict(
          (name, list(map(int, (xmin, ymin, xmax, ymax))))
          for name, xmin, ymin, xmax, ymax in names_to_bboxes)

    # Retrieve mapping from filename to variant
    variant_trainval_path = os.path.join(src_path, 'data',
                                         'images_variant_trainval.txt')
    with open(variant_trainval_path, 'r') as f:
      names_to_variants = [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]

    variant_test_path = os.path.join(src_path, 'data',
                                     'images_variant_test.txt')
    with open(variant_test_path, 'r') as f:
      names_to_variants += [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]
    
    names_to_variants = dict(names_to_variants)

    # Build mapping from variant to filenames. "Variant" refers to the aircraft
    # model variant (e.g., A330-200) and is used as the class name in the
    # dataset. The position of the class name in the concatenated list of
    # training, validation, and test class name constitutes its class ID.
    variants_to_names = collections.defaultdict(list)
    for name, variant in names_to_variants.items():
      variants_to_names[variant].append(name)

    for class_id, class_name in enumerate(all_classes):
        if not os.path.exists(os.path.join(dst_path, class_name)):
            os.makedirs(os.path.join(dst_path, class_name))
        print(f"Creating dataset for class {class_name}...")
        class_files = [
            os.path.join(src_path, 'data', 'images',
                        '{}.jpg'.format(filename))
            for filename in sorted(variants_to_names[class_name])
        ]
        bboxes = [
            names_to_bboxes[name]
            for name in sorted(variants_to_names[class_name])
        ]
        fileid = 1
        for i, path in enumerate(class_files):
            file_name = 'file_'+'0'*(6-len(str(fileid)))+str(fileid)+'.png'
            fileid += 1
            bbox = bboxes[i]
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert(mode='RGB')
                image_crop = image.crop(bbox)
                image_crop.save(os.path.join(dst_path, class_name, file_name))

def process_omniglot(src_path, dst_path):
    """change color"""
    folders = ["images_background", "images_evaluation"]
    for folder in folders:
        root_src = os.path.join(src_path,folder)
        root_dst = os.path.join(dst_path,folder)
        for superclass in os.listdir(root_src):
            for class_name in os.listdir(os.path.join(root_src,superclass)):
                if not os.path.exists(os.path.join(root_dst, superclass, class_name)):
                    os.makedirs(os.path.join(root_dst, superclass, class_name))
                for file_name in os.listdir(os.path.join(root_src,superclass,class_name)):
                    image = Image.open(os.path.join(root_src,superclass,class_name,file_name)).convert('RGB')
                    image = 255-np.array(image)
                    image = Image.fromarray(image)
                    image.save(os.path.join(root_dst,superclass,class_name, file_name))

def process_traffic_signs(src_path, dst_path):
    """change ppm to png"""
    folder = "Final_Training/Images"
    for class_name in os.listdir(os.path.join(src_path, folder)):
        if not os.path.exists(os.path.join(dst_path, class_name)):
            os.makedirs(os.path.join(dst_path, class_name))
        for file_name in os.listdir(os.path.join(src_path, folder, class_name)):
            if "ppm" in file_name:
                image = Image.open(os.path.join(src_path,folder,class_name,file_name)).convert('RGB')
                image.save(os.path.join(dst_path, class_name, file_name[:-3])+"png")   

def main():
    parser = argparse.ArgumentParser('Prepare datasets for Meta-Dataset.', add_help=False)
    parser.add_argument('--data_src_path', type=str, required=True, help="Directory of downloaded raw dataset.")
    parser.add_argument('--data_dst_path', type=str, required=True, help="Directory of processed dataset.")
    parser.add_argument('--process_mnist', type=int, default=0, help="Whether process MNIST dataset.")
    parser.add_argument('--process_CIFAR10', type=int, default=0, help="Whether process CIFAR10 dataset.")
    parser.add_argument('--process_CIFAR100', type=int, default=0, help="Whether process CIFAR100 dataset.")
    parser.add_argument('--process_DuickD', type=int, default=0, help="Whether process Quick Draw dataset.")
    parser.add_argument('--process_coco', type=int, default=0, help="Whether process MSCOCO dataset.")
    parser.add_argument('--process_aircraft', type=int, default=0, help="Whether process Aircraft dataset.")
    parser.add_argument('--process_omniglot', type=int, default=0, help="Whether process Omniglot dataset.")
    parser.add_argument('--process_traffic', type=int, default=0, help="Whether process Traffic Signs dataset.")

    args, unparsed = parser.parse_known_args()

    if  args.process_mnist:
        print('Processing MNIST test set.')
        process_mnist(args.data_src_path, args.data_dst_path)

    if  args.process_CIFAR10:
        print('Processing CIFAR10 test set.')
        process_cifar(
            src_path=os.path.join(args.data_src_path, 'cifar-10-batches-py'),
            dst_path=os.path.join(args.data_dst_path, 'cifar10'),
            imagesFileName='test_batch',
            labelsFileName='batches.meta',
            labelKey1=b'labels',
            labelKey2=b'label_names'
        )

    if  args.process_CIFAR100:
        print('Processing CIFAR100 test set.')
        process_cifar(
            src_path=os.path.join(args.data_src_path, 'cifar-100-python'),
            dst_path=os.path.join(args.data_dst_path, 'cifar100'),
            imagesFileName='test',
            labelsFileName='meta',
            labelKey1=b'fine_labels',
            labelKey2=b'fine_label_names'
        )
    
    if args.process_DuickD:
        print('Processing Quick Draw dataset.')
        process_quickdraw(src_path=args.data_src_path, dst_path=args.data_dst_path)

    if args.process_coco:
        print('Processing MSCOCO dataset.')
        process_coco(src_path=args.data_src_path, dst_path=args.data_dst_path)
    
    if args.process_aircraft:
        print('Processing Aircraft dataset.')
        process_aircraft(src_path=args.data_src_path, dst_path=args.data_dst_path)

    if args.process_omniglot:
        print('Processing Omniglot dataset.')
        process_omniglot(src_path=args.data_src_path, dst_path=args.data_dst_path)

    if args.process_traffic:
        print('Processing Traffic Signs dataset.')
        process_traffic_signs(src_path=args.data_src_path, dst_path=args.data_dst_path) 

if __name__ == '__main__':
    main()
    
