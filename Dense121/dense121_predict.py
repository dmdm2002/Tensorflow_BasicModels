import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

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

label = ['fake', 'live']
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
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = train_datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=batchsz)

input_shape = (224, 224, 3)

###########################################################################################
#                                       model
###########################################################################################

model_d = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = model_d.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.5)(x)

preds = tf.keras.layers.Dense(2, activation='softmax')(x) #FC-layer

model = tf.keras.Model(inputs=model_d.input, outputs=preds)

for layer in model.layers[:-8]:
    layer.trainable = False

for layer in model.layers[-8:]:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = '../training/densnet121/warsaw/gan_fake_war_2fold/ckp-{epoch:04d}.ckpt'
ckp_dir = os.path.dirname(ckp_path)

latest = tf.train.latest_checkpoint(ckp_dir)
print(ckp_dir)
model.load_weights(latest)

###########################################################################################
#                                       predict
###########################################################################################

model.evaluate(train_generator, verbose=1)