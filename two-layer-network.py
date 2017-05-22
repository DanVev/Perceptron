import numpy as np
from mnist import MNIST

np.random.seed(1)
IMAGE_SHAPE = (28, 28)
HIDDEN_LAYER_NEURON_NUMBER = 20
TRAINING_SPEED = 0.50
EPOCH_NUMBER = 1

print("Load data from MNIST database")
mndata = MNIST(r"D:\Projects\PycharmProjects\Perceptron\mnist")
tr_images, tr_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


def sigmoid(x, deriv=False):
    if deriv is True:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


print("Normalization proccess")
# Image Normalization
for i in range(0, len(test_images)):
    test_images[i] = np.array(test_images[i]) / 255

for i in range(0, len(tr_images)):
    tr_images[i] = np.array(tr_images[i]) / 255
print("Normalization is completed")

print("Start of training")
for neuron_number in range(5, 250, 3):
    print("--------------------------------------")
    print("Neuron number in hidden layer #", neuron_number)
    HIDDEN_LAYER_NEURON_NUMBER = neuron_number
    w1 = (2 * np.random.rand(HIDDEN_LAYER_NEURON_NUMBER, 784) - 1) / 10
    w2 = (2 * np.random.rand(10, HIDDEN_LAYER_NEURON_NUMBER) - 1) / 10
    b1 = (2 * np.random.rand(HIDDEN_LAYER_NEURON_NUMBER) - 1) / 10
    b2 = (2 * np.random.rand(10) - 1) / 10
    # TRAINING_SPEED -= 0.05
    for n in range(len(tr_images)):
        # if n % 10000 == 0:
        #    print("Training number:", n)
        img = tr_images[n]
        cls = tr_labels[n]

        # forward propagation
        resp1 = np.zeros(HIDDEN_LAYER_NEURON_NUMBER, dtype=np.float32)
        resp2 = np.zeros(10, dtype=np.float32)

        # first layer
        for i in range(0, HIDDEN_LAYER_NEURON_NUMBER):
            r = w1[i] * img
            r = sigmoid(np.sum(r) + b1[i])
            resp1[i] = r

        # second layer
        for i in range(0, 10):
            r = w2[i] * resp1
            r = sigmoid(np.sum(r) + b2[i])
            resp2[i] = r

        # class definition
        resp_cls = np.argmax(resp2)
        # resp2 = np.zeros(10, dtype=np.float32)
        # resp2[resp_cls] = 1.0

        # back propagation
        true_resp = np.zeros(10, dtype=np.float32)
        true_resp[cls] = 1.0

        error = true_resp - resp2

        delta2 = error * resp2 * (1 - resp2)

        delta1 = resp1 * (1 - resp1) * (np.dot(delta2, w2))

        for i in range(0, 10):
            w2[i] += TRAINING_SPEED * np.dot(resp1, delta2[i])
        for i in range(0, HIDDEN_LAYER_NEURON_NUMBER):
            w1[i] += TRAINING_SPEED * np.dot(img, delta1[i])
        b2 += TRAINING_SPEED * delta2
        b1 += TRAINING_SPEED * delta1

    total = len(test_images)
    valid = 0
    invalid = []


    def nn_calculate(img):
        resp1 = list(range(0, HIDDEN_LAYER_NEURON_NUMBER))
        resp2 = list(range(0, 10))
        # first layer
        for i in range(0, HIDDEN_LAYER_NEURON_NUMBER):
            r = w1[i] * img
            r = sigmoid(np.sum(r) + b1[i])
            resp1[i] = r

        # second layer
        for i in range(0, 10):
            r = w2[i] * resp1
            r = sigmoid(np.sum(r) + b2[i])
            resp2[i] = r
        return np.argmax(resp2)


    for i in range(0, total):
        img = test_images[i]
        predicted = nn_calculate(img)
        true = test_labels[i]
        if predicted == true:
            valid += 1
        else:
            invalid.append({"image": img, "predicted": predicted, "true": true})

    # print("--------------------------------------")
    print("Accuracy {}%".format(valid / total * 100))
    # print(invalid[:50])
