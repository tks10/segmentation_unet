from PIL import Image
import numpy as np
import glob
import os


class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True):
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot)

    def get_all_dataset(self):
        return self._data

    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)
        images_original, images_segmented = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        # Get a palette
        image_sample_palette = Image.open(paths_segmented[0])
        palette = image_sample_palette.getpalette()

        return DataSet(images_original, images_segmented, palette)

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

        return paths_original, paths_segmented

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        images_original, images_segmented = [], []
        # Load images from directory_path using generator.
        print("Loading original images", end="", flush=True)
        for image in Loader.image_generator(paths_original, init_size):
            images_original.append(image)
            if len(images_original) % 200 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)
        print("Loading segmented images", end="", flush=True)
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".", end="", flush="True")
        print(" Completed")
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)
        # Change indices which correspond to "void" from 255
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented

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
    CATEGORY = (
        "ground",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "void"
    )

    def __init__(self, images_original, images_segmented, image_palette):
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = np.asarray(images_original, dtype=np.float32)
        self._images_segmented = np.asarray(images_segmented, dtype=np.float32)
        self._image_palette = image_palette

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def length(self):
        return len(self._images)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented)

    def shuffle(self):
        list_packed = list(zip(self._images_original, self._images_segmented))
        np.random.shuffle(list_packed)
        images_original, images_segmented, labels = zip(*list_packed)
        return DataSet(images_original, images_segmented)

    def transpose_by_color(self):
        image_original = self._images_original.transpose(0, 3, 1, 2)
        image_segmented = self._images_segmented.transpose(0, 3, 1, 2)
        return DataSet(image_original, image_segmented)

    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end])

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
    data = dataset_loader.get_all_dataset()
    data.print_information()
