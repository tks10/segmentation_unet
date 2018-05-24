import argparse
import sys
import os
import tensorflow as tf
import numpy as np

from util import loader
from util import model

FLAGS = None
DIR_TARGET = "target"
DIR_OTHERS = "others"


def main(_):

    # Load training and test data
    # 訓練とテストデータを読み込みます


    # Whether or not using a GPU
    # GPUを使用するか
    gpu = FLAGS.gpu
    use_cnn = FLAGS.cnn

    print(sys.path)

    # Create a model
    # モデルの生成
    model_unet = model.UNet().model

    # Set loss function and optimizer
    # 誤差関数とオプティマイザの設定をします
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 精度の算出をします
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 1), tf.argmax(model_unet.teacher, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session
    # セッションの初期化をします
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9), device_count={'GPU': 0})
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train model
    # モデルの訓練
    epochs = 200
    batch_size = 32

    '''
    train_images = train.images if use_cnn else train.images_reshaped
    test_images = test.images if use_cnn else test.images_reshaped

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size):
            # バッチデータの展開
            batch_images = batch.images if use_cnn else batch.images_reshaped
            batch_labels = batch.labels
            # Back Prop
            sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels})
        # Evaluation
        # 評価
        if epoch % 10 == 0:
            loss_train = sess.run(cross_entropy, feed_dict={x: train_images, y_: train.labels})
            loss_test = sess.run(cross_entropy, feed_dict={x: test_images, y_: test.labels})
            accuracy_train = sess.run(accuracy, feed_dict={x: train_images, y_: train.labels})
            accuracy_test = sess.run(accuracy, feed_dict={x: test_images, y_: test.labels})
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
    '''

    sess.run(train_step, feed_dict={model_unet.inputs: np.zeros((1, 256, 256, 3)),
                                    model_unet.teacher: np.zeros((1, 256, 256, 2))})

    # Test trained model
    # 訓練済みモデルの評価
    loss_test = sess.run(cross_entropy, feed_dict={model_unet.inputs: np.zeros((1, 256, 256, 3)),
                                                   model_unet.teacher: np.zeros((1, 256, 256, 2))})
    accuracy_test = sess.run(accuracy, feed_dict={model_unet.inputs: np.zeros((1, 256, 256, 3)),
                                                  model_unet.teacher: np.zeros((1, 256, 256, 2))})
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images', help='Directory for storing input data')
    parser.add_argument('--cnn', action='store_true', help='Use cnn model')
    parser.add_argument('--gpu', action='store_true', help='Use gpu')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)