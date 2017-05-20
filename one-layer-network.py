import numpy as np
from mnist import MNIST

np.random.seed(1)
IMAGE_SHAPE = (28, 28)
w = (2 * np.random.rand(10, 784) - 1) / 10
b = (2 * np.random.rand(10) - 1) / 10

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
for n in range(len(tr_images)):
    if n % 1000 == 0:
        print("Training number:", n)
    img = tr_images[n]
    cls = tr_labels[n]
    # forward propagation
    resp = np.zeros(10, dtype=np.float32)
    for i in range(0, 10):
        r = w[i] * img
        r = sigmoid(np.sum(r) + b[i])
        resp[i] = r

    resp_cls = np.argmax(resp)
    # resp = np.zeros(10, dtype=np.float32)
    # resp[resp_cls] = 1.0

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

    return np.argmax(resp)


total = len(test_images)
valid = 0
invalid = []

for i in range(0, total):
    img = test_images[i]
    predicted = nn_calculate(img)
    true = test_labels[i]
    if predicted == true:
        valid += 1
    else:
        invalid.append({"image": img, "predicted": predicted, "true": true})

print("accuracy {}".format(valid / total))
# print(invalid[:50])
