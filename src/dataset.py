from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class Dataset(object):

    def __init__(self, train_file, val_file, test_file, images_path, image_target_size):
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        self._images_path = images_path
        self._image_target_size = image_target_size
        self._set_dfs()

    def _set_dfs(self):
        self._train_df = pd.read_csv(self._train_file)
        self._val_df = pd.read_csv(self._val_file)
        self._test_df = pd.read_csv(self._test_file)

    def get_train_generator(self, batch_size, seed):
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=self._train_df,
            directory=self._images_path,
            x_col='image',
            y_col=None,
            class_mode='input',
            target_size=self._image_target_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)

        return train_generator

    def get_val_generator(self, batch_size, seed):
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=self._val_df,
            directory=self._images_path,
            x_col='image',
            y_col=None,
            class_mode='input',
            target_size=self._image_target_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)

        return val_generator
