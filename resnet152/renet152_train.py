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
path = 'Z:/Iris_dataset/nd_labeling_iris_data/innerclass'

train_path = f'{path}/CycleGAN/A'
test_path = f'{path}/CycleGAN/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data

# batchsz = 4
# traincnt = 5154
# testcnt = 5182
# valcnt = 1191

# cyclegan data
batchsz = 4
traincnt = 4550
testcnt = 5018
valcnt = 1036

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)

input_shape = (224, 224, 3)

###########################################################################################
#                                       model
###########################################################################################

print("[INFO] preparing model...")

model_res = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model_res.trainable = True

x = model_res.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(512, activation='relu')(x)

preds = tf.keras.layers.Dense(2, activation='softmax')(x) #FC-layer

model = tf.keras.Model(inputs=model_res.input, outputs=preds)

model.summary()

# opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                   set tensorboard
###########################################################################################

log_dir = "../logs/fit/nd/ResNet/2-fold-Trainable-4/gan_training"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = "Z:/backup/ckp/nd/ResNet/2-fold-Trainable-4/ganfake/ckp-{epoch:04d}.ckpt"
ckp_dir = os.path.dirname(ckp_path)

ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path, verbose=1, save_weights_only=True)

###########################################################################################
#                                       test call back
###########################################################################################

# class step_test(tf.keras.callbacks.Callback):
#
#     def __init__(self, train_generator):
#         self.generator = train_generator
#
#     def on_epoch_end(self, epoch, logs=None):
#         acc = self.model.evaluate(self.generator, verbose=1)
#         print(f'{epoch} : {acc}')
#
#
# testing = step_test(train_generator)

###########################################################################################
#                                       training
###########################################################################################

# training_hisotry = model.fit(train_generator, epochs=30, callbacks=[ckp_callback, tb_callback],
#                              steps_per_epoch=(traincnt//batchsz), validation_data=test_generator,
#                               validation_steps=(testcnt//batchsz), shuffle=True)
#
# model.evaluate(test_generator, verbose=1)

###########################################################################################
#                                     training-2fold
###########################################################################################

training_hisotry = model.fit(test_generator, epochs=30, callbacks=[ckp_callback, tb_callback],
                             steps_per_epoch=(testcnt//batchsz), validation_data=train_generator,
                              validation_steps=(traincnt//batchsz), shuffle=True)

model.evaluate(train_generator, verbose=1)