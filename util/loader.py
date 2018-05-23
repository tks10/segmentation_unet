from PIL import Image
import numpy as np
import glob
import os


class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True):
        self._data = Loader.import_images(dir_original, dir_segmented, init_size, one_hot)

    @staticmethod
    def import_images(dir_original, dir_segmented, init_size=None, one_hot=True):
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

        images_original, images_segmented = [], []
        # Load images from directory_path using generator.
        print("Loading original images", end="", flush=True)
        for image in Loader.image_generator(paths_original, init_size):
            images_original.append(image)
            if len(images_original) % 200 == 0:
                print(".", end="", flush=True)
        print("Completed.", flush=True)
        print("Loading segmented images...", end="", flush="True")
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".", end="", flush="True")
        print("Completed.")
        assert len(images_original) == len(images_segmented)

        if one_hot:
            _labels = np.zeros((len(_images), label_count))
            for i in range(_labels.shape[0]):
                _labels[i][label] = 1
        else:
            _labels = np.full(len(_images), label)

        return DataSet(_images, _labels)

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True):
        """
        `A generator which yields images deleted an alpha channel and resized.
         アルファチャネル削除、リサイズ(任意)処理を行った画像を返します
        Args:
            file_paths (list[string]): File paths you want load.
            init_size (tuple(int, int)): If having a value, images are resized by init_size.
        Yields:
            image (ndarray[width][height][channel]): processed image.
        """
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                image = Image.open(file_path)
                # to square
                image = Loader.crop_to_square(image)
                # resize
                if init_size is not None and init_size != image.size:
                    image = image.resize(init_size)
                # delete alpha channel
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                    # TODO(tks10): Deal with an alpha channel.
                    # If original pixel's values aren't 255, contrary to expectations, the pixels may be not white.
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                yield image

    @staticmethod
    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))


class DataSet(object):
    BACKGROUND = (0, 0, 0)
    VOID = (255, 255, 255)

    def __init__(self, images_original, images_segmented, labels):
        assert len(images_original) == len(images_segmented) and len(images_original) == len(labels),\
            "images and labels must have same length."
        self._images_original = np.asarray(images_original, dtype=np.float32)
        self._images_segmented = np.asarray(images_segmented, dtype=np.float32)
        self._labels = np.asarray(labels, dtype=np.int32)

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def labels(self):
        return self._labels

    @property
    def length(self):
        return len(self._images)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        labels = np.concatenate([self.labels, other.labels])
        return DataSet(images_original, images_segmented, labels)

    def shuffle(self):
        list_packed = list(zip(self._images_original, self._images_segmented, self._labels))
        np.random.shuffle(list_packed)
        images_original, images_segmented, labels = zip(*list_packed)
        return DataSet(images_original, images_segmented, labels)

    def transpose_by_color(self):
        image_original = self._images_original.transpose(0, 3, 1, 2)
        image_segmented = self._images_segmented.transpose(0, 3, 1, 2)
        return DataSet(image_original, image_segmented, self._labels)

    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._labels[start:end])

    def __call__(self, batch_size=20, shuffle=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
         バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        data = self.shuffle() if shuffle else self

        for start in range(0, self.length, batch_size):
            permed = data.perm(start, start+batch_size)
            yield permed


if __name__ == "__main__":
    dataset_loader = Loader(dir_original="../data_set/VOCdevkit/VOC2012/JPEGImages",
                            dir_segmented="../data_set/VOCdevkit/VOC2012/SegmentationClass")
    train, test = dataset_loader.load_train_test()
    train.print_information()
    test.print_information()
