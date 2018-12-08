import numpy as np

import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from torch.autograd import Variable

# to_do: Add flexible CUDA support

class GAN(nn.Module):

    def __init__(self, generator, discriminator, optimizer_g, optimizer_d):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = optimizer_g(self.generator.parameters())
        self.discriminator_optimizer = optimizer_d(self.discriminator.parameters())

        self.batch_size = 10 # change

    def generate(self, n):

        zs = torch.randn((n, 1))
        examples = self.generator.forward(zs)
        return examples

    def decide(self, X):

        return self.discriminator.forward(X)

    def calculate_game_value(self, z_batch, X_batch):

        generated_examples = self.generator.forward(z_batch)

        fake_preds = self.decide(generated_examples)
        true_preds = self.decide(X_batch)

        value = torch.sum(torch.log(true_preds)) + torch.sum(torch.log(1 - fake_preds)) # add a switch for 1 - log trick which helps to avoid saturation

        value = value / (self.batch_size * 2) # taking the mean

        return value

    def calculate_generator_score(self, z_batch):
        generated_examples = self.generator.forward(z_batch)
        fake_preds = self.decide(generated_examples)

        value = -torch.sum(torch.log(fake_preds))  # add a switch for 1 - log trick which helps to avoid saturation
        value = value / (self.batch_size)  # taking the mean

        return value

    def train(self, X, n_updates):

        N = X.size(0)

        vals = []

        for i in range(n_updates):

            if not i % 1000:
                print("Update number {}".format(i))


            inds = np.random.randint(N, size=self.batch_size)
            inds = torch.LongTensor(inds)

            z_batch = Variable(torch.randn(self.batch_size, 1))
            X_batch = Variable(X[inds])

            value = -self.calculate_game_value(z_batch, X_batch)
            value.backward(retain_graph=True)

            self.discriminator_optimizer.step()

            neg_value = -value
            vals.append(neg_value)

            neg_value.backward()
            self.generator_optimizer.step()

            self.discriminator_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()

        return vals

if __name__ == "__main__":

    data = torch.randn((100, 1))

    generator = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, data.shape[1]),
    )

    discriminator = torch.nn.Sequential(
        torch.nn.Linear(data.shape[1], 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
        torch.nn.Sigmoid()
    )

    optim_generator = torch.optim.Adam
    optim_discriminator = torch.optim.Adam

    gan = GAN(generator, discriminator, optim_generator, optim_discriminator)
    vals = gan.train(data, 10000)

    plt.figure()
    xs = Variable(torch.FloatTensor(np.linspace(-5, 5, 1000).reshape(1000, 1)))
    preds = gan.decide(xs)
    plt.plot(xs.data.numpy(), preds.data.numpy())
    plt.show()

    plt.figure()
    plt.hist(gan.generate(100000).data.numpy())
    plt.show()

    plt.figure("Game Values")
    plt.plot(vals)
    plt.show()