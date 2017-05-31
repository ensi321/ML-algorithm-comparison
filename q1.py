import numpy as np
import pylab as plt
import pickle
import data
import math

# c is the number of class
c = 10
N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()

# Return num_image, num_of_ones
def q1b_helper(k, l):
	# This is array of images in training set that has label of k
	images = []
	for i, image in enumerate(train_images):
		if (train_labels[i][k] == 1):
			images.append(image)

	num_image = len(images)

	num_of_ones = 0
	for i, image in enumerate(images):
		if (image[l] >= 0.5):
			num_of_ones = num_of_ones + 1

	return num_image, num_of_ones

def save_image(theta):
	for c in range(theta.shape[0]):
		data.save_images(theta[c, :].reshape((1, theta.shape[1])), str(c) + '.jpg')


def q1b():
	# theta is a matrix c by size of image, which in this case 10 by 784
	d = len(train_images[0])
	theta = np.zeros((c, d))

	for k in range(c):
		print('c = ' + str(k))
		for l in range(d):
			print('d = ' + str(l))
			num_image, num_of_ones = q1b_helper(k, l)
			theta[k, l] = (float(num_of_ones + 1)) / (num_image + 2)

	pickle.dump(theta, open("theta.p", 'wb'))
	# save_image(theta)
	return theta

def q1c():
	pi_c = 0.1
	# theta = q1b()
	theta = pickle.load(open('theta.p', 'rb'))
	l_training = []
	l_test = []
	prediction_training = []
	prediction_test = []
	# For each image, calculate the max log likelihood for some class c, and then average them out for the entire
	# training/test set
	print('Calculating likelihood for training')
	for i, image in enumerate(train_images):
		print('i = '+ str(i))
		# A list of likelihood for each c = k
		l_list = []
		for k in range(c):
			l_list.append(calculate_likelihood(k, pi_c, image, theta))

		l_training.append(max(l_list))
		prediction_training.append(l_list.index(max(l_list)))
		print('prediction: ' + str(l_list.index(max(l_list))) + ' target: ' + str(train_labels[i]))

	avg_l_training = float(sum(l_training))/len(l_training)

	print('Calculating likelihood for test')
	for i, image in enumerate(test_images):
		print('i = '+ str(i))
		# A list of likelihood for each c = k
		l_list = []
		for k in range(c):
			l_list.append(calculate_likelihood(k, pi_c, image, theta))

		l_test.append(max(l_list))
		prediction_test.append(l_list.index(max(l_list)))
		print('prediction: ' + str(l_list.index(max(l_list))) + ' target: ' + str(test_labels[i]))

	avg_l_test = float(sum(l_test))/len(l_test)

	# Calculate the accuracy by comparing prediction and label
	num_correct = 0
	for i, prediction in enumerate(prediction_training):
		if (train_labels[i][prediction] == 1):
			num_correct = num_correct + 1

	accuracy_training = float(num_correct) / len(prediction_training)

	num_correct = 0
	for i, prediction in enumerate(prediction_test):
		if (test_labels[i][prediction] == 1):
			num_correct = num_correct + 1

	accuracy_test = float(num_correct) / len(prediction_test)

	return avg_l_training, avg_l_test, accuracy_training, accuracy_test

def calculate_likelihood(k, pi_k, image, theta):
	first_term = math.log(pi_k)
	second_term = 0
	for d in range(len(image)):
		x_d = image[d]
		theta_cd = theta[k, d]
		current_term = x_d * math.log(theta_cd) + (1 - x_d) * math.log(1 - theta_cd)
		second_term = second_term + current_term

	result = first_term + second_term

	return result

# avg_l_training, avg_l_test, accuracy_training, accuracy_test = q1c()
# print(avg_l_training)
# print(avg_l_test)
# print(accuracy_training)
# print(accuracy_test)
q1b()