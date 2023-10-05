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

# train_path = f'{path}/train_cyclegan'
# val_path = f'{path}/test_cyclegan/val'
# test_path = f'{path}/test_cyclegan/test'
train_path = f'{path}/train_crop'
val_path = f'{path}/test_crop/known'
test_path = f'{path}/test_crop/unknown'


train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data
batchsz = 4
traincnt = 4513
testcnt = 4510
valcnt = 2990

# gan fake data
# batchsz = 4
# traincnt = 3688
# testcnt = 4700
# valcnt = 1948

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

print("[INFO] preparing model...")

model_res = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model_res.trainable = False

model = tf.keras.models.Sequential()
model.add(model_res)
model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = '../training/resnet50/warsaw/original_one/ckp-{epoch:04d}.ckpt'
ckp_dir = os.path.dirname(ckp_path)

latest = tf.train.latest_checkpoint(ckp_dir)
print(ckp_dir)
model.load_weights(latest)

###########################################################################################
#                                       predict
###########################################################################################

model.evaluate(test_generator, verbose=1)