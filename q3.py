import autograd.numpy as np
from autograd import grad
import data
import pickle
from autograd.scipy.misc import logsumexp


N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
def softmax(x):
	return x - logsumexp(x, axis = 1)[: , None]


# w is a weight matrix of 10 by 784
# x is an image of N by 784
def logistic_predictions(w, x):
	# 0.1 is bias
	tmp = np.dot(x, w)
	return softmax(tmp)

#================== This is q3d ===================
# This is -log likelihood
def training_loss(w):
	preds = logistic_predictions(w, x)
	# Cross entropy
	label_probabilities = targets * preds
	return -np.sum(label_probabilities)



x = train_images
targets = train_labels

# Define a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.zeros((784, 10))

print "Initial loss:", training_loss(weights)
for i in xrange(10000):
	if (i % 100 == 0):
		print(i)
	weights -= training_gradient_fun(weights) * 0.3

print  "Trained loss:", training_loss(weights)

pickle.dump(weights, open('weights.p', 'wb'))
# Plot w
for i in range(10):
	w_i = weights[:, i]
	w_i = w_i.reshape((1, 784))
	data.save_images(w_i, str(i) + '.jpg')


def q3e():
	w = pickle.load(open('weights.p', 'rb'))
	train_prediction = logistic_predictions(w, train_images)
	l_train = np.average(np.sum(train_prediction, axis = 1))
	# print(train_prediction[0])
	train_prediction = np.argmax(train_prediction, axis = 1)
	# print(train_prediction[0])

	test_prediction = np.dot(test_images, w)
	l_test = np.average(np.sum(test_prediction, axis = 1))
	test_prediction = np.argmax(test_prediction, axis = 1)

	num_correct = 0
	for i, prediction in enumerate(train_prediction):
		print('Predict: ' + str(prediction) + ' label: ' + str(train_labels[i]))
		if (train_labels[i][prediction] == 1):
			num_correct = num_correct + 1
	train_accuracy = float(num_correct) / len(train_labels)

	num_correct = 0
	for i, prediction in enumerate(test_prediction):
		if (test_labels[i][prediction] == 1):
			num_correct = num_correct + 1
	test_accuracy = float(num_correct) / len(test_labels)

	print(train_accuracy)
	print(test_accuracy)
	print(l_train)
	print(l_test)

q3e()