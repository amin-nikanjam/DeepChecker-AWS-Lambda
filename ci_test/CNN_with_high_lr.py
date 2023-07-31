import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras import datasets, layers, models, losses, optimizers, metrics
   
from refactored_DC.checkers import DeepChecker
import refactored_DC.interfaceData as interfaceData
import refactored_DC.data as data


class Model:
    def __init__(self, X, Y):
        self.features = X
        self.n_classes = 10
        self.labels = Y 
        self.problem_type = "classification"
        self.reg_loss = tf1.losses.get_regularization_loss()
        self.loss_fct = losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = optimizers.SGD(learning_rate=0.5)
        self.perf = metrics.Accuracy()


    def build(self, l1=0.0, l2=1e-5):
            model = models.Sequential()
            model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2), input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
            model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2)))
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
            model.add(layers.Flatten())
            model.add(layers.Dense(units=1024,
                                    activation='relu',
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2)))
            model.add(layers.Dense(units=self.n_classes,
                                   activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2)))
            model.add(layers.Activation(activation=tf.nn.softmax))
            return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model = Model(x_train, y_train)
    data_under_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    
    checker = DeepChecker(name='base_CNN_high_lr', data=data_under_test, model=model, buffer_scale=10)
    checker.run_full_checks()