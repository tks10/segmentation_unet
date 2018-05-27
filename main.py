import argparse
import sys
import os
import random
import tensorflow as tf
import numpy as np
from PIL import Image

from util import loader as ld
from util import model

FLAGS = None
DIR_TARGET = "target"
DIR_OTHERS = "others"


def load_dataset():
    loader = ld.Loader(dir_original="data_set/VOCdevkit/VOC2012/JPEGImages",
                       dir_segmented="data_set/VOCdevkit/VOC2012/SegmentationClass")
    return loader.load_train_test(shuffle=False)


def get_concat(im1, im2, palette, mode):
    if mode == "P":
        dst = Image.new("P", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        dst.putpalette(palette)
    elif mode == "RGB":
        dst = Image.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    return dst


def cast_to_aaa(nlist, palette):
    assert len(nlist.shape) == 3
    res = np.argmax(nlist, axis=2)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    return image


def show_imageset(image_in_np, image_out_np, image_tc_np, palette):
    image_out, image_tc = cast_to_aaa(image_out_np, palette), cast_to_aaa(image_tc_np, palette)
    image_concated = get_concat(image_out, image_tc, palette, "P").convert("RGB")
    get_concat(Image.fromarray(np.uint8(image_in_np * 255), mode="RGB"), image_concated, None, "RGB").show()


def main(_):

    # Load training and test data
    # 訓練とテストデータを読み込みます
    train, test = load_dataset()
    valid = train.perm(0, 10)
    test = test.perm(0, 150)
    print(test.images_original.shape)

    # Whether or not using a GPU
    # GPUを使用するか
    gpu = FLAGS.gpu
    gpu = True

    # Create a model
    # モデルの生成
    model_unet = model.UNet().model

    # Set loss function and optimizer
    # 誤差関数とオプティマイザの設定をします
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 精度の算出をします
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session
    # セッションの初期化をします
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=True, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train model
    # モデルの訓練
    epochs = 2000
    batch_size = 32
    train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size):
            # バッチデータの展開
            inputs = batch.images_original
            teacher = np.float32(batch.images_segmented)
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})
        # Evaluation
        # 評価

        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            if epoch % 3 == 0:
                idx = random.randrange(10)
                outputs = sess.run(model_unet.outputs, feed_dict={model_unet.inputs: [train.images_original[idx]],
                                                                  model_unet.is_training: False})
                show_imageset(train.images_original[idx], outputs[0], train.images_segmented[idx], train.palette)

    # Test trained model
    # 訓練済みモデルの評価
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images', help='Directory for storing input data')
    parser.add_argument('--gpu', action='store_true', help='Use gpu')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)