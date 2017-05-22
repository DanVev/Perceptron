import numpy as np
from mnist import MNIST

np.random.seed(1)
IMAGE_SHAPE = (28, 28)
TRAINING_SPEED = 0.50
EPOCH_NUMBER = 5
w = (2 * np.random.rand(10, 784) - 1) / 10
b = (2 * np.random.rand(10) - 1) / 10

print("Load data from MNIST database")
mndata = MNIST(r"D:\Projects\PycharmProjects\Perceptron\mnist")
tr_images, tr_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
resp = np.zeros(10, dtype=np.float32)


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
for epoch in range(EPOCH_NUMBER):
    print("Epoch number #", epoch + 1)
    for n in range(len(tr_images)):
        # if n % 1000 == 0:
        #    print("Training number:", n)
        img = tr_images[n]
        cls = tr_labels[n]

        # forward propagation
        for i in range(0, 10):
            r = w[i] * img
            r = sigmoid(np.sum(r) + b[i])
            resp[i] = r

        resp_cls = np.argmax(resp)

        # back propagation
        true_resp = np.zeros(10, dtype=np.float32)
        true_resp[cls] = 1.0

        error = true_resp - resp

        delta = error * resp * (1 - resp)
        for i in range(0, 10):
            w[i] += np.dot(img, delta[i])
        b += delta


def nn_calculate(img):
    resp = list(range(0, 10))
    for i in range(0, 10):
        r = w[i] * img
        r = sigmoid(np.sum(r) + b[i])
        resp[i] = r

    return np.argmax(resp), resp


total = len(test_images)
valid = 0
invalid = []
pairs = {}
for i in range(0, total):
    img = test_images[i]
    predicted, resp = nn_calculate(img)
    true = test_labels[i]
    if predicted == true:
        valid += 1
    else:
        invalid.append({"predicted": (predicted, resp[predicted]), "true": (true, resp[true])})
        pairs[(predicted, true)] = pairs.get((predicted, true),0) + 1

print("accuracy {}".format(valid / total))
for string in invalid:
    print(string)
for pair, n in sorted(list(pairs.items()), key = lambda x:x[1], reverse = True):
    print(pair, n)
