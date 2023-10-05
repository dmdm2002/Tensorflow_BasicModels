import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd

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
path = 'Z:/Iris_dataset/nd_labeling_iris_data'

train_path = f'{path}/blur/2-fold/A'
test_path = f'{path}/CycleGAN/1-fold/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data
#
# batchsz = 4
# traincnt = 5154
# testcnt = 5182
# valcnt = 1191

# cyclegan data
batchsz = 2
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

model_d = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_d.trainable = True
x = model_d.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(1024, activation='relu')(x)
# x = tf.keras.layers.Dense(512, activation='relu')(x)

preds = tf.keras.layers.Dense(2, activation='softmax')(x) #FC-layer

model = tf.keras.Model(inputs=model_d.input, outputs=preds)

# for layer in model.layers[:-8]:
#     layer.trainable = False
#
# for layer in model.layers[-8:]:
#     layer.trainable = True

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                       set ckp
###########################################################################################

# ckp_path = "E:/backup/ckp/nd/Densenet169/ganfake/2-fold/ckp-{epoch:04d}.ckpt"
# ckp_dir = os.path.dirname(ckp_path)
#
# print(ckp_dir)
#
# latest = tf.train.latest_checkpoint(ckp_dir)
# print(latest)
# print(ckp_dir)
# model.load_weights(latest)

###########################################################################################
#                                       predict
###########################################################################################

for epoch in range(1, 31):
    # real_epoch = epoch + 1
    ckp_path = f"Z:/backup/ckp/nd/Densenet169/1-fold-Trainable/ganfake/ckp-{epoch:04d}.ckpt"
    model.load_weights(ckp_path)
    history = model.evaluate(test_generator, verbose=1)
    print(f'{epoch} : {history}')

# predict_array = model.predict(test_generator, verbose=1)
#
# df = pd.DataFrame()
# for i in range(0, len(predict_array)):
#     label = np.argmax(test_generator[i][1])
#     predict_label = np.argmax(predict_array[0])
#     df_temp = pd.DataFrame({'label': label, 'predict_label': predict_label})
#     df = pd.concat([df, df_temp], axis=0, ignore_index=True)
#
# df.to_csv('predict_1fold.csv')