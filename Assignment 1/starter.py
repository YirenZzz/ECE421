import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget



def loss(W, b, x, y, reg):
    # Your implementation here
    N = np.shape(y)[0]
    z = np.matmul(x,W) + b
    y_hat= 1.0/(1.0+np.exp(-z))
    
    loss_ce= (np.sum(-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))))
    loss_reg=reg/2*np.sum(W*W)
    loss_total=loss_ce+loss_reg
    return loss_total


def grad_loss(W, b, x, y, reg):
    # Your implementation here
    N = np.shape(y)[0]
    
    z = np.matmul(x,W) + b
    y_hat= 1.0/(1.0+np.exp(-1.0*z))    
    
    grad_loss_w=np.matmul(np.transpose(x), (y_hat - y))/(np.shape(y)[0]) + reg*W
    grad_loss_b=(np.sum(y_hat - y)) / N    
    return grad_loss_w, grad_loss_b


def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, vali, vali_target, test, test_target):
    # Your implementation here
    
    train_loss_array=[]
    train_acc_array=[]
    train_loss_array.append(loss(W, b, x, y, reg))
    train_acc_array.append(accuracy(W, b, x, y))
    
    validation_loss_array=[]
    validation_acc_array=[]
    validation_loss_array.append(loss(W, b, vali, vali_target, reg))
    validation_acc_array.append(accuracy(W, b, vali, vali_target))    
    
    test_loss_array=[]
    test_acc_array=[]
    test_loss_array.append(loss(W, b, test, test_target, reg))
    test_acc_array.append(accuracy(W, b, test, test_target))     
    for epoch in range(epochs):
        grad_loss_w, grad_loss_b = grad_loss(W, b, x, y, reg)
        W_new = W - alpha * grad_loss_w
        b_new = b - alpha * grad_loss_b
        
        #calculate loss
        train_loss_array.append(loss(W_new, b_new, x, y, reg))
        #add accuracy to array
        train_acc_array.append(accuracy(W_new, b_new, x, y))    
        
        
        #calculate loss
        validation_loss_array.append(loss(W_new, b_new, vali, vali_target, reg))
        #add accuracy to array
        validation_acc_array.append(accuracy(W_new, b_new, vali, vali_target))
        
        #calculate loss
        test_loss_array.append(loss(W_new, b_new, test, test_target, reg))
        #add accuracy to array
        test_acc_array.append(accuracy(W_new, b_new, test, test_target))        
        
        if(np.linalg.norm(W_new - W) < error_tol):
            break
        
        W = W_new
        b = b_new
        
    
    print("When alpha = ", alpha,"and reg=", reg)
    print("Training Accuracy", train_acc_array[-1])
    print("Validation Accuracy", validation_acc_array[-1])
    print("Test Accuracy", test_acc_array[-1])
   
    return W_new, b_new, train_loss_array, train_acc_array, validation_loss_array, validation_acc_array, test_loss_array, test_acc_array


def accuracy(W, b, x, y):
    N = np.shape(y)[0]
    z = np.matmul(x,W) + b
    y_hat= 1.0/(1.0+np.exp(-z))
    
    accuracy = np.sum( (y_hat >= 0.5) ==y ) / N 
    return accuracy

def buildGraph(beta1, beta2, epsilon,learning_rate ):
    W = tf.Variable(tf.truncated_normal([784, 1],mean=0.0, stddev=0.5, dtype=tf.float32))
    b = tf.Variable(tf.zeros(1))

    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.float32, [None, 1], name="y")
    reg = tf.placeholder(tf.float32, name="reg")
    tf.set_random_seed(421)

    logits = (tf.matmul(x, W) + b)
    loss_total=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)) + reg/2 * tf.nn.l2_loss(W)
    if (beta1!=0 and beta2==0 and epsilon==0):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=beta1).minimize(loss_total)
    elif(beta1==0 and beta2!=0 and epsilon==0):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta2=beta2).minimize(loss_total)
    elif(beta1==0 and beta2==0 and epsilon!=0):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=epsilon).minimize(loss_total)
    elif(beta1==0 and beta2==0 and epsilon==0):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_total)    
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss_total)
    return x, y, W, b, reg, loss_total, optimizer



def SGD(batchSize, trainData, trainTarget, beta1, beta2, epsilon, learning_rate,epochs):
    N=3500
    x, y, W, b, reg, loss_total, optimizer = buildGraph(beta1, beta2, epsilon,learning_rate)
    batch = N // batchSize

    train_loss_array=[]
    train_acc_array=[]    
    validation_loss_array=[]
    validation_acc_array=[]
    test_loss_array=[]
    test_acc_array=[]    
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess: 
        sess.run(init)
        for i in range(epochs):
            index = np.arange(N)
            np.random.shuffle(index)
            trainData = trainData[index]
            trainTarget = trainTarget[index]
        
            for j in range(batch):
                batch_x = trainData[j*batchSize:(j+1)*batchSize, :]
                batch_y = trainTarget[j*batchSize:(j+1)*batchSize, :]
                _, train_W, train_b = sess.run([optimizer, W, b], feed_dict={x: batch_x, y: batch_y, reg: 0})
        
            
            train_acc_array.append(accuracy(train_W, train_b, trainData, trainTarget))
            train_loss_array.append(sess.run(loss_total, feed_dict={x: trainData, y: trainTarget, reg: 0}))
            
            validation_acc_array.append(accuracy(train_W, train_b, validData, validTarget))
            validation_loss_array.append(sess.run(loss_total, feed_dict={x: validData, y: validTarget, reg: 0}))
            
            
            test_acc_array.append(accuracy(train_W, train_b, testData, testTarget))
            test_loss_array.append(sess.run(loss_total, feed_dict={x: testData, y: testTarget, reg: 0}))
            
            
    print('Batch size:',batchSize)
    if (beta1!=0 and beta2==0 and epsilon==0):
        print('beta1:',beta1)
    elif(beta1==0 and beta2!=0 and epsilon==0):
        print('beta2:',beta2)
    elif(beta1==0 and beta2==0 and epsilon!=0):
        print('epsilon:',epsilon)
   
    print("Trainning Loss",train_loss_array[-1])
    print("Validation Loss",validation_loss_array[-1])
    print("Test Loss",test_loss_array[-1])    
    print("Training Accuracy", train_acc_array[-1])
    print("Validation Accuracy", validation_acc_array[-1])
    print("Test Accuracy", test_acc_array[-1])
    

    return train_loss_array, train_acc_array, validation_loss_array, validation_acc_array, test_loss_array, test_acc_array
    
     
    

#testing 
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#Reshape trainData to pass into functions

trainData=trainData.reshape(3500,784)
validData = validData.reshape(100,784)
testData = testData.reshape(145,784)
W= np.random.normal(0,0.5,(trainData.shape[1],1))
b=0

#trainData = trainData.reshape(trainData.shape[0], -1)
#validData = validData.reshape(validData.shape[0], -1)
#testData = testData.reshape(testData.shape[0], -1)

#================================Part 1======================================
#part 1 Q3 tuning the learning rate
#initialize parameters
epochs = 5000
alpha=[0.005, 0.001, 0.0001]
reg = 0
error_tol = 1e-7


#when alpha=0.005, plot train loss and validation loss--------------------------
W_train, b_train,train_loss_array1, train_acc_array1,val_loss_array1, val_acc_array1, test_loss_array1, test_acc_array1  = grad_descent(W, b, trainData, trainTarget, alpha[0], epochs, reg, error_tol, validData, validTarget,testData, testTarget)

plt.figure(1)
plt.plot(train_loss_array1, label='training')
plt.plot(val_loss_array1, label='validation')
plt.plot(test_loss_array1, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with alpha=0.005")
plt.legend(loc='best')


#when alpha=0.005, plot train accuracy and validation accuracy------------------
plt.figure(2)
plt.plot(train_acc_array1, label='training')
plt.plot(val_acc_array1, label='validation')
plt.plot(test_acc_array1, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with alpha=0.005")
plt.legend(loc='best')


#-------------------------------------------------------------------------------

#when alpha=0.001, plot train loss and validation loss--------------------------
W_train, b_train,train_loss_array2, train_acc_array2,val_loss_array2, val_acc_array2, test_loss_array2, test_acc_array2  = grad_descent(W, b, trainData, trainTarget, alpha[1], epochs, reg, error_tol, validData, validTarget,testData, testTarget)

plt.figure(3)
plt.plot(train_loss_array2, label='training')
plt.plot(val_loss_array2, label='validation')
plt.plot(test_loss_array2, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with alpha=0.001")
plt.legend(loc='best')


#when alpha=0.001, plot train accuracy and validation accuracy------------------
plt.figure(4)
plt.plot(train_acc_array2, label='training')
plt.plot(val_acc_array2, label='validation')
plt.plot(test_acc_array2, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with alpha=0.001")
plt.legend(loc='best')

#-------------------------------------------------------------------------------
#when alpha=0.0001, plot train accuracy and validation accuracy------------------
W_train, b_train,train_loss_array3, train_acc_array3,val_loss_array3, val_acc_array3, test_loss_array3, test_acc_array3  = grad_descent(W, b, trainData, trainTarget, alpha[2], epochs, reg, error_tol, validData, validTarget,testData, testTarget)

plt.figure(5)
plt.plot(train_loss_array3, label='training')
plt.plot(val_loss_array3, label='validation')
plt.plot(test_loss_array3, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with alpha=0.0001")
plt.legend(loc='best')


#when alpha=0.0001, plot train accuracy and validation accuracy------------------
plt.figure(6)
plt.plot(train_acc_array3, label='training')
plt.plot(val_acc_array3, label='validation')
plt.plot(test_acc_array3, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with alpha=0.0001")
plt.legend(loc='best')

#===========================================================================================================
#part 1 Q4 Generalization
epochs = 5000
reg = [0.001, 0.1, 0.5]
alpha=0.005
error_tol = 1e-7

#-------------------------------------------------------------------------------
#when reg=0.001, plot train loss and validation loss--------------------------

W_train, b_train,train_loss_array4, train_acc_array4,val_loss_array4, val_acc_array4, test_loss_array4, test_acc_array4  = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[0], error_tol, validData, validTarget,testData, testTarget)

plt.figure(7)
plt.plot(train_loss_array4, label='training')
plt.plot(val_loss_array4, label='validation')
plt.plot(test_loss_array4, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with reg=0.001")
plt.legend(loc='best')


#when reg=0.001, plot train accuracy and validation accuracy------------------
plt.figure(8)
plt.plot(train_acc_array4, label='training')
plt.plot(val_acc_array4, label='validation')
plt.plot(test_acc_array4, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with reg=0.001")
plt.legend(loc='best')

#-------------------------------------------------------------------------------

#when reg=0.1, plot train loss and validation loss--------------------------
W_train, b_train,train_loss_array5, train_acc_array5,val_loss_array5, val_acc_array5, test_loss_array5, test_acc_array5  = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[1], error_tol, validData, validTarget,testData, testTarget)

plt.figure(9)
plt.plot(train_loss_array5, label='training')
plt.plot(val_loss_array5, label='validation')
plt.plot(test_loss_array5, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with reg=0.1")
plt.legend(loc='best')


#when reg=0.1, plot train accuracy and validation accuracy------------------
plt.figure(10)
plt.plot(train_acc_array5, label='training')
plt.plot(val_acc_array5, label='validation')
plt.plot(test_acc_array5, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with reg=0.1")
plt.legend(loc='best')

#-------------------------------------------------------------------------------
#when reg=0.5, plot train accuracy and validation accuracy------------------
W_train, b_train,train_loss_array6, train_acc_array6,val_loss_array6, val_acc_array6, test_loss_array6, test_acc_array6  = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[2], error_tol, validData, validTarget,testData, testTarget)

plt.figure(11)
plt.plot(train_loss_array6, label='training')
plt.plot(val_loss_array6, label='validation')
plt.plot(test_loss_array6, label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with reg=0.5")
plt.legend(loc='best')


#when reg=0.5, plot train accuracy and validation accuracy------------------
plt.figure(12)
plt.plot(train_acc_array6, label='training')
plt.plot(val_acc_array6, label='validation')
plt.plot(test_acc_array6, label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with reg=0.5")
plt.legend(loc='best')

#============================Part2================================================
epochs=700
beta1 =[0,0.95, 0.99]
beta2 = [0,0.99, 0.9999]
epsilon = [0,1e-09, 1e-4]
batchSize=[500, 100, 700, 1750]
learning_rate=0.001

#Q2 minibatch_size=500, epochs=700, reg=0, alpha=0.001 ----------------------
train_loss_array7, train_acc_array7,validation_loss_array7, validation_acc_array7, test_loss_array7, test_acc_array7=SGD(batchSize[0], trainData, trainTarget, beta1[0], beta2[0], epsilon[0], learning_rate, epochs)

plt.figure(13)
plt.plot(train_loss_array7, label='Training')
plt.plot(validation_loss_array7, label='Validation')
plt.plot(test_loss_array7, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500")
plt.legend(loc='best')

plt.figure(14)
plt.plot(train_acc_array7, label='Training')
plt.plot(validation_acc_array7, label='Validation')
plt.plot(test_acc_array7, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500")
plt.legend(loc='best')   

#Q3 epochs=700, reg=0, alpha=0.001 ----------------------
#when minibatch_size=100 -----------------------
train_loss_array8, train_acc_array8,validation_loss_array8, validation_acc_array8, test_loss_array8, test_acc_array8=SGD(batchSize[1], trainData, trainTarget, beta1[0], beta2[0], epsilon[0], learning_rate, epochs)

plt.figure(15)
plt.plot(train_loss_array8, label='Training')
plt.plot(validation_loss_array8, label='Validation')
plt.plot(test_loss_array8, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=100")
plt.legend(loc='best')

plt.figure(16)
plt.plot(train_acc_array8, label='Training')
plt.plot(validation_acc_array8, label='Validation')
plt.plot(test_acc_array8, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=100")
plt.legend(loc='best')  

train_loss_array9, train_acc_array9,validation_loss_array9, validation_acc_array9, test_loss_array9, test_acc_array9=SGD(batchSize[2], trainData, trainTarget, beta1[0], beta2[0], epsilon[0], learning_rate, epochs)

#when minibatch_size=700 -----------------------
plt.figure(17)
plt.plot(train_loss_array9, label='Training')
plt.plot(validation_loss_array9, label='Validation')
plt.plot(test_loss_array9, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=700")
plt.legend(loc='best')

plt.figure(18)
plt.plot(train_acc_array9, label='Training')
plt.plot(validation_acc_array9, label='Validation')
plt.plot(test_acc_array9, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=700")
plt.legend(loc='best')  


#when minibatch_size=1750 -----------------------
train_loss_array10, train_acc_array10,validation_loss_array10, validation_acc_array10, test_loss_array10, test_acc_array10=SGD(batchSize[3], trainData, trainTarget, beta1[0], beta2[0], epsilon[0], learning_rate, epochs)
plt.figure(19)
plt.plot(train_loss_array10, label='Training')
plt.plot(validation_loss_array10, label='Validation')
plt.plot(test_loss_array10, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=1750")
plt.legend(loc='best')

plt.figure(20)
plt.plot(train_acc_array10, label='Training')
plt.plot(validation_acc_array10, label='Validation')
plt.plot(test_acc_array10, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=1750")
plt.legend(loc='best')  


#Q4 epochs=700, reg=0, alpha=0.001, batchSize=500 ----------------------
#when beta1=0.95, beta2=default, epsilon=default-------------
train_loss_array11, train_acc_array11,validation_loss_array11, validation_acc_array11, test_loss_array11, test_acc_array11=SGD(batchSize[0], trainData, trainTarget, beta1[1], beta2[0], epsilon[0], learning_rate, epochs)
plt.figure(21)
plt.plot(train_loss_array11, label='Training')
plt.plot(validation_loss_array11, label='Validation')
plt.plot(test_loss_array11, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and beta1=0.95")
plt.legend(loc='best')

plt.figure(22)
plt.plot(train_acc_array11, label='Training')
plt.plot(validation_acc_array11, label='Validation')
plt.plot(test_acc_array11, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and beta1=0.95")
plt.legend(loc='best') 

#when beta1=0.99, beta2=default, epsilon=default----------
train_loss_array12, train_acc_array12,validation_loss_array12, validation_acc_array12, test_loss_array12, test_acc_array12=SGD(batchSize[0], trainData, trainTarget, beta1[2], beta2[0], epsilon[0], learning_rate, epochs)
plt.figure(23)
plt.plot(train_loss_array12, label='Training')
plt.plot(validation_loss_array12, label='Validation')
plt.plot(test_loss_array12, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and beta1=0.99")
plt.legend(loc='best')

plt.figure(24)
plt.plot(train_acc_array12, label='Training')
plt.plot(validation_acc_array12, label='Validation')
plt.plot(test_acc_array12, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and beta1=0.99")
plt.legend(loc='best')


#when beta1=default, beta2=0.99, epsilon=default----------
train_loss_array13, train_acc_array13,validation_loss_array13, validation_acc_array13, test_loss_array13, test_acc_array13=SGD(batchSize[0], trainData, trainTarget, beta1[0], beta2[1], epsilon[0], learning_rate, epochs)
plt.figure(25)
plt.plot(train_loss_array13, label='Training')
plt.plot(validation_loss_array13, label='Validation')
plt.plot(test_loss_array13, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and beta2=0.99")
plt.legend(loc='best')

plt.figure(26)
plt.plot(train_acc_array13, label='Training')
plt.plot(validation_acc_array13, label='Validation')
plt.plot(test_acc_array13, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and beta2=0.99")
plt.legend(loc='best')


#when beta1=default, beta2=0.9999, epsilon=default----------
train_loss_array14, train_acc_array14,validation_loss_array14, validation_acc_array14, test_loss_array14, test_acc_array14=SGD(batchSize[0], trainData, trainTarget, beta1[0], beta2[2], epsilon[0], learning_rate, epochs)
plt.figure(27)
plt.plot(train_loss_array14, label='Training')
plt.plot(validation_loss_array14, label='Validation')
plt.plot(test_loss_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and beta2=0.9999")
plt.legend(loc='best')

plt.figure(28)
plt.plot(train_acc_array14, label='Training')
plt.plot(validation_acc_array14, label='Validation')
plt.plot(test_acc_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and beta2=0.9999")
plt.legend(loc='best')

#when beta1=default, beta2=default, epsilon=1e-09----------
train_loss_array14, train_acc_array14,validation_loss_array14, validation_acc_array14, test_loss_array14, test_acc_array14=SGD(batchSize[0], trainData, trainTarget, beta1[0], beta2[0], epsilon[1], learning_rate, epochs)
plt.figure(29)
plt.plot(train_loss_array14, label='Training')
plt.plot(validation_loss_array14, label='Validation')
plt.plot(test_loss_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and epsilon=1e-09")
plt.legend(loc='best')

plt.figure(30)
plt.plot(train_acc_array14, label='Training')
plt.plot(validation_acc_array14, label='Validation')
plt.plot(test_acc_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and epsilon=1e-09")
plt.legend(loc='best')

#when beta1=default, beta2=default, epsilon=1e-4----------
train_loss_array14, train_acc_array14,validation_loss_array14, validation_acc_array14, test_loss_array14, test_acc_array14=SGD(batchSize[0], trainData, trainTarget, beta1[0], beta2[0], epsilon[2], learning_rate, epochs)
plt.figure(31)
plt.plot(train_loss_array14, label='Training')
plt.plot(validation_loss_array14, label='Validation')
plt.plot(test_loss_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Loss")    
plt.title("Loss vs Epochs with batch size=500 and epsilon=1e-4")
plt.legend(loc='best')

plt.figure(32)
plt.plot(train_acc_array14, label='Training')
plt.plot(validation_acc_array14, label='Validation')
plt.plot(test_acc_array14, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")    
plt.title("Accuracy vs Epochs with batch size=500 and epsilon=1e-4")
plt.legend(loc='best')

plt.show()
