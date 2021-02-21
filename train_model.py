#importing the libraries like keras, sklearn and matplotlib
from keras.optimizers import SGD   
from keras.preprocessing.image import ImageDataGenerator  
import matplotlib.pyplot as plt
import LeNet
from sklearn.metrics import classification_report

# Loading images from the directory in batches of 64
traingen=ImageDataGenerator(rescale=1./255,rotation_range=30,horizontal_flip=True,fill_mode='nearest')
testgen=ImageDataGenerator(rescale=1./255)
train_generator=traingen.flow_from_directory(r'C:\Users\91945\Downloads\Analysis\DL.ai\Internship\T2\data\train set',class_mode='categorical',batch_size=64,target_size=(128,128),shuffle=True)
test_generator=testgen.flow_from_directory(r'C:\Users\91945\Downloads\Analysis\DL.ai\Internship\T2\data\val set',class_mode='categorical',batch_size=64,target_size=(128,128),shuffle=True)

# Building my CNN model
lenet=LeNet.LeNet(128,128,3,2)
model1=lenet.build()
model1.summary()

# Fine tuning the optimizer
opt=SGD(lr=0.01,decay=1e-5,momentum=0.9)

# Compiling my CNN model according to my data
model1.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

H1=model1.fit_generator(train_generator,epochs=30,validation_data=test_generator)
pred=model1.predict(test_generator)
model1.save("model1.hdf5")

# Plotting the accuracy loss graph with respect to epochs
plt.style.use("ggplot")
hist=H1.history
plt.plot(hist['accuracy'],label='accuracy')
plt.plot(hist['val_accuracy'],label='val_accuracy')
plt.plot(hist['loss'],label='loss')
plt.plot(hist['val_loss'],label='val_loss')
plt.title("GRAPHS")
plt.xlabel("epochs")
plt.ylabel("acc/loss")
plt.legend()
plt.savefig("graph.png")
plt.show()
















