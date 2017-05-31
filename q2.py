import numpy as np
import pylab as plt
import pickle
import data
import math
# import q1
import random


N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
theta = pickle.load(open('theta.p', 'rb'))

def q2c():
	theta = q1.q1b()
	# theta = pickle.load(open('theta.p', 'rb'))
	pi_c = 0.1

	for i in range(10):
		# Random generate c
		c = random.randint(0, 9)
		# Generate x using uniform distribution
		# x is a 1 x 784
		x = np.random.rand(1, 784)
		theta_c = (np.asarray(theta[c, :])).reshape((1, 784))
		# Compare uniform generated c to theta
		# if x_d >= theta_cd, image_d = 1, else image_d = 0
		tmp = x - theta_c
		tmp = tmp >= 0
		image = tmp.astype(int)

		print(x)
		print(theta_c)
		print(image)
		data.save_images(image.reshape((1, theta.shape[1])), str(i) + '.jpg')

# Pick the first 20 images from training set
def q2f():
	for i in range(20):
		print(i)
		image = train_images[i]
		top = image[:392]
		bottom = []


		for d in range(392,784):
			# Calculate p(x = 0| x_top , theta) and p(x = 1| x_top, theta)
			p_0 = 0
			p_1 = 0
			for k in range(10):
				product_1 = theta[k, d]
				product_0 = (1 - theta[k, d])

				product = 1
				# This is p(x_t | k, theta)
				for d_prime in range(0, 392):
					product = product * math.pow(theta[k, d_prime], top[d_prime]) * math.pow((1 - theta[k, d_prime]), (1 - top[d_prime]))

				p_1 = p_1 + (product_1 * product)
				p_0 = p_0 + (product_0 * product)


			# x_d = 1 if p(x = 1 | x_top, theta) >= p(x = 0 | x_top, theta)
			print('p_0: ' + str(p_0))
			print('p_1: ' + str(p_1))
			x_d = 1 if p_1 >= p_0 else 0
			bottom.append(x_d)

		img = top.tolist()  + bottom
		print('=============================')
		print(img)
		img = np.asarray(img)
		print(img.shape)
		data.save_images(img.reshape((1, 784)), str(i) + 'b.jpg')


q2f()