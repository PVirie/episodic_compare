import torch
import numpy as np
import cv2
from dataset import FashionMNIST
from episodic import Episodic
from compare import Compare


if __name__ == "__main__":
    print("main")

    device = torch.device("cuda:0")

    batch_size = 1
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=1000, seed=0, group_size=2)

    comparator = Compare(device)
    episodic = Episodic(device)

    percent_correct = 0.0
    for i, (data, label) in enumerate(dataset):
        print("data: ", i)

        input = data.to(device)
        output = label.to(device)

        if i >= 1:
            a = episodic.retrieve_all()
            indices = comparator.compare(input, a)
            prediction = episodic.resolve(indices[:, 0])
            prediction_cpu = prediction.cpu()
            correct = (prediction == output)
            count_correct = np.sum(correct.cpu().numpy())
            percent_correct = 0.99 * percent_correct + 0.01 * count_correct * 100 / batch_size
            print("Truth: ", dataset.readout(label))
            print("Guess: ", dataset.readout(prediction_cpu.flatten()))
            print("Percent correct: ", percent_correct)

        episodic.learn(input, output)
        p, n = episodic.sample(output, 10, 10)
        comparator.learn(input, p, n, steps=1)

        img = np.reshape(data.numpy(), [-1, data.shape[2]])
        cv2.imshow("sample", img)
        cv2.waitKey(10)

    print("Computing backward scores...")
    count = 0
    for i, (data, label) in enumerate(dataset):
        input = data.to(device)
        output = label.to(device)

        # test
        a = episodic.retrieve_all()
        indices = comparator.compare(input, a)
        prediction = episodic.resolve(indices[:, 0]).cpu()
        count = count + np.sum(prediction.numpy() == label.numpy())

    print("Percent correct: ", count * 100 / (len(dataset) * batch_size))
