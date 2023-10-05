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

label = ['Fake', 'Live']
path = 'Z:/Iris_dataset/nd_labeling_iris_data'

train_path = f'{path}/blur/2-fold/A'
test_path = f'{path}/blur_99/1-fold/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data

# batchsz = 4
# traincnt = 5154
# testcnt = 5182
# valcnt = 1191

# cyclegan data
batchsz = 2
traincnt = 4554
testcnt = 5018
valcnt = 1036

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz, shuffle=False)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz, shuffle=False)

input_shape = (224, 224, 3)

###########################################################################################
#                                       model
###########################################################################################

# , input_tensor=tf.keras.layers.Input(shape=(224, 224, 3))
model_vgg = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model_vgg.trainable = False

x = model_vgg.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
preds = tf.keras.layers.Dense(2, activation='softmax')(x) #FC-layer

model = tf.keras.Model(inputs=model_vgg.input, outputs=preds)

# model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                       set ckp
###########################################################################################

# ckp_path = "E:/backup/ckp/nd/VGG19/CycleGAN/1-fold/ckp-{epoch:04d}.ckpt"
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

for epoch in range(30, 31):
    # real_epoch = epoch + 1
    ckp_path = f'Z:/backup/ckp/nd/VGG19/2-fold-nonTrainable/ckp-{epoch:04d}.ckpt'
    model.load_weights(ckp_path)
    history = model.evaluate(train_generator, verbose=1)
    print(f'{epoch} : {history}')
