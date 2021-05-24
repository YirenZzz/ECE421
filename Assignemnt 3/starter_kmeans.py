import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp




# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    X_expand = tf.expand_dims(X, 0)
    MU_expand = tf.expand_dims(MU, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(X_expand, MU_expand)), 2)
    return distances
    

def loss_function(distance):
    return tf.reduce_sum(tf.reduce_min(distance, axis=0))


def k_mean(k, D):
    tf.set_random_seed(421)
    X = tf.placeholder(tf.float32, shape=[None, D])
    means = tf.Variable(tf.random.normal(shape=[k, D]), name="means")
    distance=distanceFunc(X, means)
    
    loss = tf.reduce_sum(tf.reduce_min(distance, axis=0))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
    
    return X, means, distance, loss, optimizer


def train_k_mean(trainData, k, iterations):
    D=train_data.shape[1]
    X, means, distance, loss, optimizer =k_mean(k, D)
    
    train_lossArr=[]
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            train_loss, _ = sess.run([loss, optimizer], feed_dict={X: trainData})
            train_lossArr.append(train_loss)
            
        mu = means.eval()      
        cluster = sess.run(tf.argmin(distance, 0), feed_dict={X: trainData})
     
    percentages = np.zeros(k)
    str_percentages=[]
    for i in range(k):
        percentages[i] = np.sum(np.equal(i, cluster))*100.0/len(cluster)
        str_percentages.append(str(percentages[i])+'%')
        print("cluster", i)
        print("mean:", mu[i])
        print("percentage:", percentages[i],"%")
        print()


    print("Final loss:", train_lossArr[-1])
    scatter=plt.scatter(data[:, 0], data[:, 1], c=cluster, s=25, alpha=0.6)
    plt.plot(mu[:, 0], mu[:, 1], 'kx', markersize=10)
    plt.title(str(k)+ '-Means Clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=str_percentages)

    plt.figure()
    plt.title(str(k)+ '-Means Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_lossArr)
    
    plt.show()



def k_mean_holdout(trainData, validData, k, iterations):
    D=trainData.shape[1]
    X, means, distance, loss, optimizer =k_mean(k, D)
    train_lossArry=[]
    loss_vals=[]
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            loss_train, _ = sess.run([loss, optimizer], feed_dict={X: trainData})
            train_lossArry.append(loss_train)
            
            valid_loss = sess.run(loss, feed_dict={X: validData}) 
            loss_vals.append(valid_loss)

        mean_vals = means.eval()      
        membership_vals = sess.run(tf.argmin(distance, 0), feed_dict={X: trainData})
    
    percentages = np.zeros(k)
    str_percentages=[]
    for i in range(k):
        percentages[i] = np.sum(np.equal(i, membership_vals))*100.0/len(membership_vals)
        str_percentages.append(str(round(percentages[i],2))+'%')
        print("cluster", i)
        print("mean:", mean_vals[i])
        print("percentage:", percentages[i],"%")
        print()

    print('K = ', k)
    print("train loss:", train_lossArry[-1])
    print('validation loss: ', loss_vals[-1])

    scatter=plt.scatter(data[:, 0], data[:, 1], c=membership_vals, s=25, alpha=0.6)
    plt.plot(mean_vals[:, 0], mean_vals[:, 1], 'kx', markersize=10)
    plt.title(str(k)+ '-Means Clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=str_percentages)

    figure1=plt.figure()
    plt.title(str(k)+ '-Means Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_lossArry, label="training")
    plt.plot(loss_vals, label="validation")
    plt.legend(loc="best")
    figure1.text(.5, -0.05, "Validation loss: "+str(loss_vals[-1]), ha='center')
    
    plt.show()
    
def k_mean_holdout_loss(trainData, validData, k, iterations):
    D=trainData.shape[1]
    X, means, distance, loss, optimizer =k_mean(k, D)
    train_lossArry=[]
    loss_vals=[]
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            loss_train, _ = sess.run([loss, optimizer], feed_dict={X: trainData})
            train_lossArry.append(loss_train)
            
            valid_loss = sess.run(loss, feed_dict={X: validData}) 
            loss_vals.append(valid_loss)

        mean_vals = means.eval()      
        membership_vals = sess.run(tf.argmin(distance, 0), feed_dict={X: trainData})
    
    percentages = np.zeros(k)
    str_percentages=[]
    print('K = ', k)
    for i in range(k):
        percentages[i] = np.sum(np.equal(i, membership_vals))*100.0/len(membership_vals)
        str_percentages.append(str(round(percentages[i],2))+'%')
        print("cluster", i)
        print("percentage:", percentages[i],"%")
        print()

    print("train loss:", train_lossArry[-1])
    print('validation loss: ', loss_vals[-1])

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
train_k_mean(data, k=3, iterations=200)


is_valid = True

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]
    
k_mean_holdout(data, val_data, k=1, iterations=200)
k_mean_holdout(data, val_data, k=2, iterations=200)
k_mean_holdout(data, val_data, k=3, iterations=200)
k_mean_holdout(data, val_data, k=4, iterations=200)
k_mean_holdout(data, val_data, k=5, iterations=200)


# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]
    
k_mean_holdout_loss(data,val_data, k=5, iterations=400)
k_mean_holdout_loss(data,val_data, k=10, iterations=400)
k_mean_holdout_loss(data,val_data, k=15, iterations=400)
k_mean_holdout_loss(data,val_data, k=20, iterations=400)
k_mean_holdout_loss(data,val_data, k=30, iterations=400)
