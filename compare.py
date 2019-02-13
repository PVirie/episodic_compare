import torch


def criterion(p, s):
    return torch.mean(-torch.exp(-(p - s)**2), dim=2)


def resnet_weights(device, c_in, kernel_size, c_out=None):

    if c_out is None:
        c_out = c_in * 2

    w0 = torch.empty(c_out, c_in, 1, 1, device=device, requires_grad=True)
    torch.nn.init.xavier_normal_(w0)
    b0 = torch.zeros(c_out, device=device, requires_grad=True)

    w1 = torch.empty(c_out, c_out, kernel_size[0], kernel_size[1], device=device, requires_grad=True)
    torch.nn.init.xavier_normal_(w1)
    b1 = torch.zeros(c_out, device=device, requires_grad=True)

    w2 = torch.empty(c_out, c_in, kernel_size[0], kernel_size[1], device=device, requires_grad=True)
    torch.nn.init.xavier_normal_(w2)
    b2 = torch.zeros(c_out, device=device, requires_grad=True)

    weights = [w0, b0, w1, b1, w2, b2]
    return weights


def resnet_block(input, stride, padding, w0, b0, w1, b1, w2, b2):

    x = input
    a = torch.nn.functional.conv2d(x, w2, b2, stride=stride, padding=padding)
    x = torch.nn.functional.conv2d(x, w0, b0, stride=1)
    x = torch.nn.functional.elu(x)
    x = torch.nn.functional.conv2d(x, w1, b1, stride=stride, padding=padding)
    x = x + a
    out = torch.nn.functional.elu(x)
    return out


class Compare:

    def __init__(self, device, layers=3, kernel_size=(3, 3), stride=(2, 2), file_path=None):
        print("init")
        self.device = device
        self.file_path = file_path
        self.weights = []
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size[0] // 2, kernel_size[0] // 2)

        self.weights.extend(resnet_weights(self.device, 1, self.kernel_size, 4))
        self.weights.extend(resnet_weights(self.device, 4, self.kernel_size, 8))
        self.weights.extend(resnet_weights(self.device, 8, self.kernel_size, 16))
        self.weights.extend(resnet_weights(self.device, 16, self.kernel_size, 32))

    def __internal__forward(self, input):
        batch = input.shape[0]

        x = input
        for i in range(len(self.weights) // 6):
            x = resnet_block(x, self.stride, self.padding, *(self.weights[(6 * i):(6 * (i + 1))]))

        return torch.reshape(x, [batch, -1])

    def save(self):
        if self.file_path:
            torch.save({"weights": self.weights}, self.file_path)

    def load(self):
        if self.file_path:
            temp = torch.load(self.file_path)
            self.weights = temp["weights"]

    # input = [batch, c, h, w]
    # positives = [batch, count, c, h, w]
    # negative = [batch, count, c, h, w]
    def learn(self, input, positives, negatives, lr=1e-4, steps=10, verbose=False):

        b = input.shape[0]
        count = positives.shape[1]
        c = input.shape[1]
        h = input.shape[2]
        w = input.shape[3]

        optimizer = torch.optim.Adam(self.weights, lr=lr)

        for i in range(steps):

            pivot = torch.unsqueeze(self.__internal__forward(input), 1).repeat(1, count, 1)
            h_pos = torch.reshape(self.__internal__forward(torch.reshape(positives, [-1, c, h, w])), [b, count, -1])
            h_neg = torch.reshape(self.__internal__forward(torch.reshape(negatives, [-1, c, h, w])), [b, count, -1])

            loss_pos = torch.mean(criterion(pivot, h_pos))
            loss_neg = torch.mean(criterion(pivot, h_neg))

            loss = loss_pos - loss_neg + 1.0

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if verbose and i % 100 == 0:
                print("loss:", loss.item())
        if verbose:
            print("loss:", loss.item())

    def compare(self, input, smple, k=1):

        batch = input.shape[0]
        count = smple.shape[0]

        x = torch.unsqueeze(self.__internal__forward(input), 1).repeat(1, count, 1)
        y = torch.unsqueeze(self.__internal__forward(smple), 0).repeat(batch, 1, 1)

        scores = criterion(x, y)
        _, indices = torch.topk(scores, k, dim=1, largest=False)
        return indices


if __name__ == '__main__':
    print("assert that the compare class works.")

    dtype = torch.float
    device = torch.device("cuda:0")

    comparator = Compare(device)

    x = torch.randn(2, 1, 28, 28, device=device)
    p = torch.randn(2, 5, 1, 28, 28, device=device)
    n = torch.randn(2, 5, 1, 28, 28, device=device)
    test = torch.cat([p[:, 0, ...], n[:, 0, ...]], dim=0)

    results = comparator.compare(x, test)
    print("before learned results:", results)

    comparator.learn(x, p, n, steps=1000, verbose=True)

    results = comparator.compare(x, test)
    print("after learned results:", results)
