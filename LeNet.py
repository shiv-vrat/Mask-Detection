from keras.layers import Dense,Flatten,Activation,MaxPool2D,Conv2D
from keras import Model,Input

class LeNet:
    def __init__(self,width,height,depth,classes):
        self.width=width
        self.height=height
        self.depth=depth
        self.classes=classes
    def build(self):
        input_=Input(shape=(self.height,self.width,self.depth))
        layer1=Conv2D(kernel_size=(5,5),filters=20,padding="same")(input_)
        layer1=Activation(activation="relu")(layer1)
        layer2=MaxPool2D(pool_size=(2,2),strides=(2,2))(layer1)
        layer3=Conv2D(kernel_size=(5,5),filters=50,padding="same")(layer2)
        layer3=Activation(activation="relu")(layer3)
        layer4=MaxPool2D(pool_size=(2,2),strides=(2,2))(layer3)
        layer4=Flatten()(layer4)
        layer5=Dense(units=512)(layer4)
        layer5=Activation(activation="relu")(layer5)
        layer6=Dense(units=self.classes)(layer5)
        layer6=Activation(activation="softmax")(layer6)
        model=Model(inputs=input_,outputs=layer6)
        return model
        