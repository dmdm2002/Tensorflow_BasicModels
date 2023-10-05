import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime

###########################################################################################
#                                        set gpu
###########################################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate xGB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def gen_itertest(iris_generator, iris_upper_generator, iris_lower_generator):
    for i in range(len(iris_generator)):
        iris = iris_generator.next()
        iris_upper = iris_upper_generator.next()
        iris_lower = iris_lower_generator.next()

        yield [iris[0], iris_upper[0], iris_lower[0]], iris[1]  # Twice the input, only once the label
###########################################################################################
#                                       load dataset
###########################################################################################

label = ['Fake', 'Live']
path = 'E:/Iris_dataset/Warsaw_labeling_iris_data/'

A_path = 'innerclass/Proposed/A/'
B_path = 'innerclass/Proposed/B/'

iris_path = 'iris'
iris_upper_path = 'iris_upper'
iris_lower_path = 'iris_lower'

A_iris = f'{path}{A_path}{iris_path}'
A_iris_upper = f'{path}{A_path}{iris_upper_path}'
A_iris_lower = f'{path}{A_path}{iris_lower_path}'

B_iris = f'{path}{B_path}{iris_path}'
B_iris_upper = f'{path}{B_path}{iris_upper_path}'
B_iris_lower = f'{path}{B_path}{iris_lower_path}'

# original data

# batchsz = 4
# traincnt = 6054
# testcnt = 5959
# valcnt = 1191

# cyclegan data
batchsz = 1
traincnt = 5154
testcnt = 5182
valcnt = 1036

A_datagen = ImageDataGenerator(rescale=1./255)
A_iris_generator = A_datagen.flow_from_directory(A_iris, target_size=(224, 224), batch_size=batchsz, shuffle=False)
A_iris_upper_generator = A_datagen.flow_from_directory(A_iris_upper, target_size=(224, 224), batch_size=batchsz, shuffle=False)
A_iris_lower_generator = A_datagen.flow_from_directory(A_iris_lower, target_size=(224, 224), batch_size=batchsz, shuffle=False)

B_datagen = ImageDataGenerator(rescale=1./255)
B_iris_generator = B_datagen.flow_from_directory(B_iris, target_size=(224, 224), batch_size=batchsz, shuffle=False)
B_iris_upper_generator = B_datagen.flow_from_directory(B_iris_upper, target_size=(224, 224), batch_size=batchsz, shuffle=False)
B_iris_lower_generator = B_datagen.flow_from_directory(B_iris_lower, target_size=(224, 224), batch_size=batchsz, shuffle=False)

# val_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz, subset='validation')
# A_iris_full_dataset = gen_itertest(A_iris_generator, A_iris_upper_generator, A_iris_lower_generator)
# B_iris_full_dataset = gen_itertest(B_iris_generator, B_iris_upper_generator, B_iris_lower_generator)

A_full_dataset = gen_itertest(A_iris_generator, A_iris_upper_generator, A_iris_lower_generator)
B_full_dataset = gen_itertest(B_iris_generator, B_iris_upper_generator, B_iris_lower_generator)



input_shape = (224, 224, 3)

###########################################################################################
#                                       model
###########################################################################################

model_res = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model_res.trainable = False

x = model_res.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
embed = tf.keras.layers.Dense(64, activation="softmax")(x)

model_iris = tf.keras.Model(inputs=model_res.inputs, outputs=embed)
model_iris_upper = tf.keras.Model(inputs=model_res.inputs, outputs=embed)
model_iris_lower = tf.keras.Model(inputs=model_res.inputs, outputs=embed)

iris = tf.keras.layers.Input(input_shape, name='iris')
iris_upper = tf.keras.layers.Input(input_shape, name='iris_upper')
iris_lower = tf.keras.layers.Input(input_shape, name='iris_lower')

encoded_iris = model_iris(iris)
encoded_upper = model_iris_upper(iris_upper)
encoded_lower = model_iris_lower(iris_lower)

fusion = tf.keras.layers.Concatenate()([encoded_iris, encoded_upper, encoded_lower])

prediction = tf.keras.layers.Dense(2, activation='softmax')(fusion)

iris_model = tf.keras.Model(inputs=[iris, iris_upper, iris_lower],
                            outputs=prediction,
                            )

opt = tf.keras.optimizers.Adam(0.001)

iris_model.compile(loss="categorical_crossentropy", optimizer=opt)

iris_model.summary()

# tf.keras.utils.plot_model(iris_model, "multi_input_and_output_model.png", show_shapes=True)

###########################################################################################
#                                   set tensorboard
###########################################################################################

log_dir = "../logs/fit/iris_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_1-fold"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = "E:/backup/ckp/iris_model/1-fold/ckp-{epoch:04d}.ckpt"
ckp_dir = os.path.dirname(ckp_path)

ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path, verbose=1, save_weights_only=True)

###########################################################################################
#                                       test call back
###########################################################################################


class step_test(tf.keras.callbacks.Callback):

    def __init__(self):
        self.data = B_full_dataset

    def on_epoch_end(self, epoch, logs=None):
        acc = self.iris_model.evaluate(self.data, verbose=1)
        print('--------------------test--------------------')
        print(f'{epoch + 1} : {acc}')
        print('--------------------------------------------')


testing = step_test()


###########################################################################################
#                                       training\
###########################################################################################

training_hisotry = iris_model.fit_generator(A_full_dataset, epochs=50, callbacks=[ckp_callback, tb_callback],
                                            steps_per_epoch=(traincnt//batchsz), shuffle=True)

# training_hisotry.evaluate({"iris":B_iris_generator, "iris_upper":B_iris_upper_generator, "iris_lower":B_iris_lower_generator}, verbose=1)

###########################################################################################
#                                     training-2fold
###########################################################################################
#
# training_hisotry = model.fit(test_generator, epochs=20, callbacks=[ckp_callback, tb_callback, testing],
#                              steps_per_epoch=(testcnt//batchsz))
#
# model.evaluate(train_generator, verbose=1)