import os
from random import shuffle

import tensorflow as tf
# from tensorflow.data import Dataset, Iterator
data = tf.data
Dataset = data.Dataset
Iterator = data.Iterator

import pandas as pd
NUM_CLASSES = 2


def read_attr_file( attr_path, image_dir):
    f = open(attr_path)
    lines = f.readlines()
    lines = list(map(lambda line: line.strip(), lines))
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = list(map(lambda line: line.split(), lines))
    df = pd.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].map(lambda x: os.path.join(image_dir, x))

    return df


def get_celebA_files(dataset_path, attribute, test=False, n_test=200):
    attr_file = os.path.join(dataset_path, 'list_attr_celebs.txt')
    image_dir = os.path.join(dataset_path, 'celebA')
    image_data = read_attr_file(attr_file, image_dir)
    # first_row = image_data.iloc[0].values

    poz_data = image_data[image_data[attribute] == '1']['image_path'].values
    neg_data = image_data[image_data[attribute] == '-1']['image_path'].values
    return poz_data, neg_data


def input_parser(img_path, label):
    # convert the label to one-hot encoding
    # one_hot = tf.one_hot(label, NUM_CLASSES)

    # read the img from file
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_resized = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 128, 128), img_decoded)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (2.0 / 255.0) * x - 1.0, img_float)
    return img_normalized, label


def get_input(dataset_path, batch_size, for_training = True):
    poz_imgs, neg_imgs = get_celebA_files(dataset_path, 'Eyeglasses')

    if for_training:
        shuffle(poz_imgs)
        shuffle(neg_imgs)

    dataset_len = min(len(poz_imgs), len(neg_imgs))

    # Get image paths
    poz_train_imgs = tf.constant(poz_imgs[:dataset_len])
    poz_train_labels = tf.constant([1] * dataset_len, dtype=tf.int32)

    neg_train_imgs = tf.constant(neg_imgs[:dataset_len])
    neg_train_labels = tf.constant([0] * dataset_len, dtype=tf.int32)

    # create TensorFlow Dataset objects
    poz_data = Dataset.from_tensor_slices((poz_train_imgs, poz_train_labels))
    neg_data = Dataset.from_tensor_slices((neg_train_imgs, neg_train_labels))

    if for_training:
        poz_data = poz_data.shuffle(dataset_len).repeat()
        neg_data = neg_data.shuffle(dataset_len).repeat()

    poz_data = poz_data.batch(batch_size)
    neg_data = neg_data.batch(batch_size)

    poz_data = poz_data.map(input_parser)
    neg_data = neg_data.map(input_parser)

    # create TensorFlow Iterator object for images with POSITIVE attribute value
    iterator_poz = Iterator.from_structure(poz_data.output_types,
                                           poz_data.output_shapes)
    # iterator_poz = poz_data.make_initializable_iterator()
    # iterator_neg = neg_data.make_initializable_iterator()
    next_element_poz = iterator_poz.get_next()

    # create TensorFlow Iterator object for images with NEGATIVE attribute value
    iterator_neg = Iterator.from_structure(neg_data.output_types,
                                           neg_data.output_shapes)
    next_element_neg = iterator_neg.get_next()

    # create two initialization ops to switch between the datasets
    poz_init_op = iterator_poz.make_initializer(poz_data)
    neg_init_op = iterator_neg.make_initializer(neg_data)
    # poz_init_op = iterator_poz.initializer
    # neg_init_op = iterator_neg.initializer

    tensor_dict = {
        "init_ops": [poz_init_op, neg_init_op],
        "next_element_poz": next_element_poz,
        "next_element_neg": next_element_neg,
    }
    return tensor_dict

def test_input():
    poz_imgs, neg_imgs = get_celebA_files('../data/', 'Eyeglasses')
    shuffle(poz_imgs)
    shuffle(neg_imgs)
    dataset_len = min(len(poz_imgs), len(neg_imgs))
    # Get image paths
    poz_train_imgs = tf.constant(poz_imgs[:dataset_len])
    poz_train_labels = tf.constant([1] * dataset_len)

    neg_train_imgs = tf.constant(neg_imgs[:dataset_len])
    neg_train_labels = tf.constant([0] * dataset_len)

    # create TensorFlow Dataset objects
    poz_data = Dataset.from_tensor_slices((poz_train_imgs, poz_train_labels)).shuffle(dataset_len).repeat()
    neg_data = Dataset.from_tensor_slices((neg_train_imgs, neg_train_labels)).shuffle(dataset_len).repeat()

    # poz_data = poz_data.map(input_parser)
    # neg_data = neg_data.map(input_parser)

    # create TensorFlow Iterator object for images with POSITIVE attribute value
    iterator_poz = Iterator.from_structure(poz_data.output_types,
                                       poz_data.output_shapes)
    next_element_poz = iterator_poz.get_next()

    # create TensorFlow Iterator object for images with NEGATIVE attribute value
    iterator_neg = Iterator.from_structure(neg_data.output_types,
                                           neg_data.output_shapes)
    next_element_neg = iterator_neg.get_next()

    # create two initialization ops to switch between the datasets
    poz_init_op = iterator_poz.make_initializer(poz_data)
    neg_init_op = iterator_neg.make_initializer(neg_data)

    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(poz_init_op)
        sess.run(neg_init_op)
        # get each element of the training dataset until the end is reached
        while True:
            try:
                elem_poz = sess.run(next_element_poz)
                elem_neg = sess.run(next_element_neg)

                print(elem_neg)
                print(elem_poz)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break


if __name__ == '__main__':
    test_input()