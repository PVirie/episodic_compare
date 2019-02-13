import torch
from dataset import FashionMNIST
import numpy as np
import cv2


def choice(input, flag, k):

    salt = flag + torch.randn_like(flag) * 1e-4
    _, indices = torch.topk(salt, min(k, salt.shape[1]), dim=1)

    shape = input.shape
    i_shape = list(indices.shape)
    one_shape = [1] * len(shape)
    e_input = input.unsqueeze(0).repeat(i_shape[0], *one_shape)
    e_indices = torch.reshape(indices, i_shape + one_shape[:-1]).repeat(1, 1, *shape[1:])

    output = torch.gather(e_input, dim=1, index=e_indices)
    return output, indices


class Episodic:
    def __init__(self, device, file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.file_path = file_path

    def learn(self, input, output):

        new_weight = (input, output)

        # merge
        self.weights.append(new_weight)

    def sample(self, sample_output, num_intra_class, num_inter_class):
        all_outputs = torch.cat([
            B for (A, B) in self.weights
        ], dim=0)

        all_samples = self.retrieve_all()

        flags = (torch.reshape(sample_output, [-1, 1]) == all_outputs).type(torch.float)

        positives, _ = choice(all_samples, flags, num_intra_class)
        negatives, _ = choice(all_samples, 1 - flags, num_inter_class)

        return positives, negatives

    def retrieve_all(self):

        all_samples = torch.cat([
            A for (A, B) in self.weights
        ], dim=0)
        return all_samples

    def resolve(self, indices):

        all_outputs = torch.cat([
            B for (A, B) in self.weights
        ], dim=0)

        return all_outputs[indices]


if __name__ == '__main__':

    dtype = torch.float
    device = torch.device("cuda:0")

    episodic = Episodic(device)

    dataset = FashionMNIST(device, batch_size=10, max_per_class=50, seed=100, group_size=1)

    for i, (data, label) in enumerate(dataset):

        input = data.to(device)
        output = label.to(device)

        episodic.learn(input, output)
        p, n = episodic.sample(output, 5, 5)

        pos = p.cpu().numpy()
        neg = n.cpu().numpy()

        cpu_input = data.numpy()

        img = np.concatenate([
            np.reshape(cpu_input, [-1, cpu_input.shape[2]]),
            np.reshape(np.transpose(pos[:, :, 0, ...], [0, 2, 1, 3]), [-1, cpu_input.shape[2] * pos.shape[1]]),
            1.0 - np.reshape(np.transpose(neg[:, :, 0, ...], [0, 2, 1, 3]), [-1, cpu_input.shape[2] * neg.shape[1]])
        ], axis=1)

        cv2.imshow("sample", img)
        cv2.waitKey(-1)
