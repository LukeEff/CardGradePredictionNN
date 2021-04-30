import os
import tensorflow as tf
import dataset as ds

def test_model(img_path):
    model_path = os.path.join('.', 'models', 'vgg16_acc_save_foobar')  # TODO change name of model to the real name
    print("load {}".format(model_path))
    model = tf.keras.models.load_model(model_path)
    test_set = ds.init_dataset(subset='test', validation_split=0, data_directory=img_path)
    predictions = model.predict(x=test_set)
    print(predictions)
