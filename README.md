# ML-algorithm-comparison
This is my solution to an assignment from csc412 - Probabilistic Learning and Reasoning

We explore the performance of generative (Naive bayes and EM algorithm) and discriminative models (Logistic regression) using MNIST.
We use mixture of Bernoulli ie. each pixel is a bernoulli random variable and they are independent of each other. We try to find which
method has a better accuracy to predict the digit of an image.


First, we fit a naive bayes model using both ML (maximum likelihood) and MAP (maximum a posteriori) to predict the digit of an image.

Secondly, we use naive bayes to predict bottom half of an image given the top half.

Thirdly, we use logistic regression along with a simple one layer neural net for gradient descent optimizer for prediction.

Lastly, we use EM along with mixture of Bernoullis.

The assignment handout is named handout.pdf
The solution, result and performances are inside solution.pdf
