import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Distance function for GMM
def distanceFunc(X, means):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # means: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    #diffs = tf.expand_dims(X, -1) - tf.transpose(means)
    #x_means2 = tf.reduce_sum(tf.square(diffs), axis=1)
    return tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - means), 2)

def log_GaussPDF(X, means, sigma):
    # Inputs
    # X: N X D
    # means: K X D
    # sigma: K X 1
    # Outputs:
    # log Gaussian PDF N X K
    # TODO
    D = tf.cast(tf.rank(X), tf.float32)
    distance = distanceFunc(X, means)
    sigma = tf.square(tf.transpose(sigma)) 

    return -0.5 * tf.log(((2 * np.pi)**D) * sigma) - distance /(2 * sigma)

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    # Outputs
    # log_post: N X K
    # TODO
    log_pi = tf.transpose(log_pi)
    return log_PDF + log_pi - hlp.reduce_logsumexp(log_PDF + log_pi, keep_dims=True)

def GMM(K, D):
	tf.set_random_seed(421)
	X = tf.placeholder(tf.float32, shape=(None, D), name="trainData")
	means = tf.Variable(tf.random_normal(shape=[K, D], stddev=1.0), name="means")
	
	phi = tf.Variable(tf.random_normal(shape=[K, 1], stddev=1.0), name="Phi")
	psi = tf.Variable(tf.random_normal(shape=[K, 1], stddev=1.0), name="Psi")
	
	sigma = tf.sqrt(tf.exp(phi))
	psi_soft = hlp.logsoftmax(psi)
	prob = tf.exp(psi_soft)

	log_gauss = log_GaussPDF(X, means, sigma)
	
	loss = - tf.reduce_sum(hlp.reduce_logsumexp(log_gauss + tf.transpose(tf.log(prob)),1),axis=0) 

	# Returns a Nx1 vector of best_cluster assignments
	best_cluster = tf.argmax(log_posterior(log_gauss, prob), axis=1)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	return X, means, prob, best_cluster, loss, optimizer, log_gauss
    
def train_GMM(trainingData, K, epochs):
    X, means, sigma, best_cluster, loss, optimizer, log_gauss = GMM(K, trainingData.shape[1])
    lossArr = []
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
      sess.run(init)
      for i in range(epochs):
          lossVal, _ =sess.run([loss, optimizer], feed_dict={X: trainingData})
          lossArr.append(lossVal)
    
      clusterCenter = means.eval()
      clusters = sess.run(best_cluster, feed_dict={X: trainingData})

    clusterAssignments = clusters.squeeze()
    percentages = np.zeros(K)
    str_percentages=[]
    for i in range(K):
        percentages[i] = np.sum(np.equal(i, clusterAssignments))*100.0/len(clusterAssignments)
        str_percentages.append(str(percentages[i])+'%')
        print("cluster", i)
        print("mean:", clusterCenter[i])
        print("percentage:", percentages[i],"%")
        print()

    print('k = ', K)
    print("Training loss:", lossArr[-1])
	
    scatter=plt.scatter(trainingData[:, 0], trainingData[:, 1], c=clusterAssignments, s=25, alpha=0.6)
    plt.plot(clusterCenter[:, 0], clusterCenter[:, 1], 'kx', markersize=10)
    plt.title(str(K)+ '-Means GMM Clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=str_percentages)
    
    plt.figure()
    plt.title(str(K)+ '-Means Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(lossArr)
    
    plt.show()
    
    
def train_GMM_holdout(trainingData, validData, K, epochs):
    X, means, sigma, best_cluster, loss, optimizer, log_gauss = GMM(K, trainingData.shape[1])
    lossArr = []
    val_lossArr = []
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
      sess.run(init)
      for i in range(epochs):
          lossVal, _ =sess.run([loss, optimizer], feed_dict={X: trainingData})
          lossArr.append(lossVal)
          valid_loss = sess.run(loss, feed_dict={X: validData})
          val_lossArr.append(valid_loss)
    
      clusterCenter = means.eval()
      clusters = sess.run(best_cluster, feed_dict={X: trainingData})
    
    clusterAssignments = clusters.squeeze()
    percentages = np.zeros(K)
    str_percentages=[]
    for i in range(K):
        percentages[i] = np.sum(np.equal(i, clusterAssignments))*100.0/len(clusterAssignments)
        str_percentages.append(str(round(percentages[i],2))+'%')
        print("cluster", i)
        print("mean:", clusterCenter[i])
        print("percentage:", percentages[i],"%")
        print()

    print('K = ', K)
    print("Training loss:", lossArr[-1])
    print("Validation loss:", val_lossArr[-1])
	
    scatter=plt.scatter(trainingData[:, 0], trainingData[:, 1], c=clusterAssignments, s=25, alpha=0.6)
    plt.plot(clusterCenter[:, 0], clusterCenter[:, 1], 'kx', markersize=10)
    plt.title(str(K)+ '-Means GMM Clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=str_percentages)
    
    figure1=plt.figure()
    plt.title(str(K)+ '-Means Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(lossArr, label="training")
    plt.plot(val_lossArr, label="validation")
    plt.legend(loc='best')
    figure1.text(.5, -0.05, "Validation loss: "+str(val_lossArr[-1]), ha='center')

    plt.show()

def train_GMM_holdout_loss(trainingData, validData, K, epochs):
    X, means, sigma, best_cluster, loss, optimizer, log_gauss = GMM(K, trainingData.shape[1])
    lossArr = []
    val_lossArr = []
    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
      sess.run(init)
      for i in range(epochs):
          lossVal, _ =sess.run([loss, optimizer], feed_dict={X: trainingData})
          lossArr.append(lossVal)
          valid_loss = sess.run(loss, feed_dict={X: validData})
          val_lossArr.append(valid_loss)
    
      clusterCenter = means.eval()
      clusters = sess.run(best_cluster, feed_dict={X: trainingData})
    
    clusterAssignments = clusters.squeeze()
    percentages = np.zeros(K)
    str_percentages=[]
    print('K = ', K)
    for i in range(K):
        percentages[i] = np.sum(np.equal(i, clusterAssignments))*100.0/len(clusterAssignments)
        str_percentages.append(str(round(percentages[i],2))+'%')
        print("cluster", i)
        print("percentage:", percentages[i],"%")
        print()

    
    print("Training loss:", lossArr[-1])
    print("Validation loss:", val_lossArr[-1])

	
'''
# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
train_GMM(data, K=3, epochs=600)
'''
# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
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

train_GMM_holdout(data,val_data, K=1, epochs=400)
train_GMM_holdout(data,val_data, K=2, epochs=400)
train_GMM_holdout(data,val_data, K=3, epochs=400)
train_GMM_holdout(data,val_data, K=4, epochs=400)
train_GMM_holdout(data,val_data, K=5, epochs=400)

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
    
# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
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
    
    
train_GMM_holdout_loss(data, val_data, K=5, epochs=400)
train_GMM_holdout_loss(data, val_data, K=10, epochs=400)
train_GMM_holdout_loss(data, val_data, K=15, epochs=400)
train_GMM_holdout_loss(data, val_data, K=20, epochs=400)    
train_GMM_holdout_loss(data, val_data, K=30, epochs=400)