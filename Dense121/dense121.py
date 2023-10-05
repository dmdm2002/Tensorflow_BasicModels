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

###########################################################################################
#                                       set dataset
###########################################################################################

label = ['Fake', 'Live']
path = 'E:/Iris dataset/Warsaw_labeling_iris_data/'

train_path = f'{path}/train_cyclegan'
val_path = f'{path}/test_cyclegan/val'
test_path = f'{path}/test_cyclegan/test'

# train_path = f'{path}/train_crop'
# val_path = f'{path}/test_crop/known'
# test_path = f'{path}/test_crop/unknown'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data

# batchsz = 4
# traincnt = 4513
# testcnt = 4510
# valcnt = 2990

# gan fake data

batchsz = 4
traincnt = 3688
testcnt = 4700
valcnt = 1948

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz, shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz, shuffle=True)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = train_datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=batchsz, shuffle=True)

input_shape = (224, 224, 3)

###########################################################################################
#                                       model
###########################################################################################

model_d = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = model_d.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.5)(x)

preds = tf.keras.layers.Dense(2, activation='softmax')(x) #FC-layer

model = tf.keras.Model(inputs=model_d.input, outputs=preds)

for layer in model.layers[:-8]:
    layer.trainable = False

for layer in model.layers[-8:]:
    layer.trainable = True

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


###########################################################################################
#                                   set tensorboard
###########################################################################################

log_dir = "../logs/fit/densenet121" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "gan_fake_war_2fold_2"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = '../training/densnet121/warsaw/gan_fake_war_2fold_2/ckp-{epoch:04d}.ckpt'
ckp_dir = os.path.dirname(ckp_path)

ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path, verbose=1, save_weights_only=True)

###########################################################################################
#                                       test call back
###########################################################################################


class step_test(tf.keras.callbacks.Callback):

    def __init__(self, train_generator):
        self.train_generator = train_generator

    def on_epoch_end(self, epoch, logs=None):
        acc = self.model.evaluate(self.train_generator, verbose=1)
        print(f'{epoch + 1} : {acc}')


testing = step_test(train_generator)


###########################################################################################
#                                       training\
###########################################################################################

# training_hisotry = model.fit(train_generator, epochs=20, callbacks=[ckp_callback, tb_callback, testing],
#                              validation_data=val_generator, steps_per_epoch=(traincnt//batchsz),
#                              validation_steps=(valcnt//batchsz))
#
# model.evaluate(test_generator, verbose=1)

###########################################################################################
#                                     training-2fold
###########################################################################################
#
training_hisotry = model.fit(test_generator, epochs=50, callbacks=[ckp_callback, tb_callback, testing],
                             validation_data=val_generator, steps_per_epoch=(testcnt//batchsz),
                             validation_steps=(valcnt//batchsz))

model.evaluate(train_generator, verbose=1)