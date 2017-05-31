import math
import numpy as np
import random
import data
import pickle



N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
x = train_images
K = 30

def q4d():
	# EM algorithm
	# Initial parameters
	pi = np.zeros(K)
	pi.fill(float(1)/K)
	print(pi)
	# Initialize theta randomly
	theta = np.random.normal(0.5, 0.1, (K, 784))

	#
	delta = 0.001
	improvement = float('inf')
	counter = 0
	# Repeat this until converges
	# while (improvement > delta):
	while (improvement > delta):
		print('Beginning of ' + str(counter))
		print(pi[0:10])
		# E-step 
		z = np.ones((len(x), K))
		for n in range(len(x)):
			# Every z_n,a has a common denominator if they share the same n
			denom = 0
			for m in range(K):
				current_product = np.prod(np.power(theta[m, :], x[n, :]) * np.power(1 - theta[m, :], 1 - x[n, :])) * pi[m]
				denom = denom + current_product

			# Calculate numerator for this specific z_n,k
			for k in range(K):
				numerator = np.prod(np.power(theta[k, :], x[n, :]) * np.power(1 - theta[k, :], 1 - x[n, :])) * pi[k]
				

				z[n, k] = numerator / denom
			if (n % 1000 == 0):
				print('n: ' + str(n))
				# print('numerator: ' + str(numerator) + ' denom: ' + str(denom))
				# print(z[n, k])


		# M-step
		# Update pi
		N = np.sum(z, axis=0)
		new_pi = N / len(x)
		# # Update theta
		new_theta = np.zeros((K, 784))
		for m in range(K):
			new_theta[m, :] = np.sum(x * z[:, m][:, np.newaxis], axis = 0) / N[m]

		improvement = max(np.average(abs(new_pi - pi)), np.average(abs(new_theta - theta)))
		pi = new_pi
		theta = new_theta
		print('Improvement: ' + str(improvement))

	pickle.dump(theta, open('theta2.p', 'wb'))
	pickle.dump(pi, open('pi2.p', 'wb'))


	theta = pickle.load(open('theta2.p', 'rb'))
	for i in range(theta.shape[0]):
		data.save_images(theta[i, :].reshape((1, 784)), str(i) + '.jpg')


def q4e():
	theta = pickle.load(open('theta2.p', 'rb'))
	pi = pickle.load(open('pi2.p', 'rb'))

	for i in range(20):
		print(i)
		image = train_images[i]
		top = image[:392]
		bottom = []


		for d in range(392,784):
			# Calculate p(x = 0| x_top , theta) and p(x = 1| x_top, theta)
			p_0 = 0
			p_1 = 0

			for k in range(K):
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

q4e()