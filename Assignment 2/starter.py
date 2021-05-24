import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    # TODO
    return np.maximum(x, 0)

def softmax(x):
    # TODO
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax    

def computeLayer(X, W, b):
    # TODO
    return np.matmul(X, W) + b


def CE(target, prediction):
    # TODO
    return -np.mean(target*np.log(prediction+1e-10))


def gradCE(target, prediction):
    N=trainData.shape[0]
    return (softmax(prediction) - target)/N

def gradWo(target, prediction, h):
    dL_do = gradCE(target, prediction)
    #do_dWo=h
    dL_dWo = np.matmul(np.transpose(h),dL_do)
    return dL_dWo

def gradbo(target, prediction):
    dL_do = gradCE(target, prediction)
    do_dbo = np.ones((target.shape[0],1))
    dL_dbo = np.matmul(np.transpose(do_dbo), dL_do)
    return dL_dbo

def gradWh(target, prediction, x, hin, Wo):
    #hin=Wh x +bh
    hin[hin <= 0] = 0
    hin[hin > 0] = 1
    dL_do = gradCE(target, prediction)
    #do_dh = Wo
    #dh_dW_h = x
    dL_dWh = np.matmul(np.transpose(x),(np.matmul(dL_do, np.transpose(Wo))*hin))
    return dL_dWh 

def gradbh(target, prediction, hin, Wo):
  #hin=Wh x +bh
    hin[hin <= 0] = 0
    hin[hin > 0] = 1
    ones = np.ones((1, hin.shape[0]))
    dL_do = gradCE(target, prediction)
    #do_dh = Wo
    #dh_dbh = 1 or 0
    dL_dbh = np.matmul(ones, (np.matmul(dL_do, np.transpose(Wo))*hin))
    return dL_dbh

def XavierInit(units_in, units_out):
    return np.random.normal(0,np.sqrt(2.0/(units_in+units_out)), (units_in, units_out))

def forward(x,Wh,bh,Wo,bo,y):
    hin=computeLayer(x, Wh, bh)
    #hidden layer: h=ReLU(Wh x + bh)
    h=relu(computeLayer(x, Wh, bh))
  
    pin=computeLayer(h, Wo, bo)
    #output layer: p=softmax(o), o=Wo h + bh
    p=softmax(computeLayer(h, Wo, bo))
  
    #prediction and accuracy calculation
    prediction= np.argmax(p, axis=1)
    accuracy= np.argmax(y, axis=1)
    return hin,h,pin,p, prediction, accuracy

#set gamma=0.99, learning rate=0.1, hidden units=1000, epoches=200

#load data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1) 
testData = testData.reshape(testData.shape[0], -1)

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

epochs=200
H=1000
gamma=0.9
alpha=0.1
F=trainData.shape[1]

#initialize weight matrices and bias vectors to zero
Wo=XavierInit(H, 10)
bo = np.zeros((1, 10))
Wh=XavierInit(F, H)
bh = np.zeros((1, H))

vWo = 1e-5 * np.ones((H, 10))
vbo = 1e-5 * np.ones((1, 10))
vWh = 1e-5 * np.ones((F, H))
vbh = 1e-5 * np.ones((1, H))

#compute a forward pass of the training data
#using the gradients derived in section 1.2, 
#implement the backpropagation algorithm to update all of the network’s weights and biases.
train_loss = []
val_loss = []
test_loss = []

train_accuracy = []
valid_accuracy = []
test_accuracy= []

for i in range(epochs):
    print("Epoch:", i)
    #train data
    hin,h,pin,p, prediction, accuracy= forward(trainData,Wh,bh,Wo,bo,trainTarget)
    train_loss.append(CE(trainTarget, p))
    train_accuracy.append(np.sum((prediction==accuracy))/(trainData.shape[0]))
    #valid data
    val_hin,val_h,val_pin,val_p, val_prediction, val_accuracy= forward(validData,Wh,bh,Wo,bo,validTarget)
    val_loss.append(CE(validTarget, val_p))
    valid_accuracy.append(np.sum((val_prediction==val_accuracy))/(validData.shape[0]))
    #test data
    test_hin,test_h,test_pin,test_p, test_prediction, test_acc= forward(testData,Wh,bh,Wo,bo,testTarget)
    test_loss.append(CE(testTarget, test_p))
    test_accuracy.append(np.sum((test_prediction==test_acc))/(testData.shape[0]))
  
    dL_dWo = gradWo(trainTarget, pin, h)
    dL_dbo = gradbo(trainTarget, pin)
    dL_dWh = gradWh(trainTarget, pin, trainData, hin, Wo)
    dL_dbh = gradbh(trainTarget, pin, hin, Wo)
  
    vWo=gamma* vWo + alpha*dL_dWo
    Wo=Wo-vWo
  
    vbo=gamma* vbo + alpha*dL_dbo
    bo=bo-vbo
  
    vWh=gamma* vWh + alpha*dL_dWh
    Wh=Wh-vWh

    vbh=gamma* vbh + alpha*dL_dbh
    bh=bh-vbh

print("training loss:", train_loss[-1])
print("valid loss:", val_loss[-1])
print("test loss:", test_loss[-1])

print("training accuracy:", train_accuracy[-1])
print("valid accuracy:", valid_accuracy[-1])
print("test accuracy:", test_accuracy[-1])


plt.figure(1)
plt.plot(train_loss,label='training')
plt.plot(val_loss,label='validation')
plt.plot(test_loss, label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Loss vs Epochs with alpha=0.1, gamma=0.9")

plt.figure(2)
plt.title("Accuracy vs Epochs with alpha=0.1, gamma=0.9")
plt.plot(train_accuracy,label='training')
plt.plot(valid_accuracy,label='validation')
plt.plot(test_accuracy, label='test')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='best')

plt.title
plt.show()
