import tensorflow as tf


class UNet:
    def __init__(self, size=(572, 572)):
        self.model = self.create_model(size)

    @staticmethod
    def create_model(size):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3])
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], 3])

        conv1_1 = UNet.conv(inputs, filters=64)
        conv1_2 = UNet.conv(conv1_1, filters=64)
        pool1 = UNet.pool(conv1_2)

        conv2_1 = UNet.conv(pool1, filters=128)
        conv2_2 = UNet.conv(conv2_1, filters=128)
        pool2 = UNet.pool(conv2_2)

        conv3_1 = UNet.conv(pool2, filters=256)
        conv3_2 = UNet.conv(conv3_1, filters=256)
        pool3 = UNet.pool(conv3_2)

        conv4_1 = UNet.conv(pool3, filters=512)
        conv4_2 = UNet.conv(conv4_1, filters=512)
        pool4 = UNet.pool(conv4_2)

        conv5_1 = UNet.conv(pool4, filters=1024)
        conv5_2 = UNet.conv(conv5_1, filters=1024)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=512), conv4_2], axis=1)

        conv_up1_1 = UNet.conv(concated1, filters=512)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=512)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=256), conv3_2], axis=1)

        conv_up2_1 = UNet.conv(concated2, filters=256)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=256)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=128), conv2_2], axis=1)

        conv_up3_1 = UNet.conv(concated3, filters=128)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=128)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=64), conv1_2], axis=1)

        conv_up4_1 = UNet.conv(concated4, filters=64)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=64)
        outputs = UNet.conv(conv_up4_2, filters=2, kernel_size=[1, 1])

        return Model(inputs, outputs, teacher)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3]):
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation=tf.nn.relu,
        )
        return conved

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters):
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=[2, 2],
            padding='valid',
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
