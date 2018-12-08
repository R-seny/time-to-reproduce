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

        self.latent_dim = 1

    def generate(self, n):

        zs = torch.randn((n, self.latent_dim))
        examples = self.generator.forward(zs)
        return examples

    def decide(self, X):

        return self.discriminator.forward(X)

    def calculate_game_value(self, z_batch, X_batch):

        generated_examples = self.generator.forward(z_batch)

        fake_preds = self.decide(generated_examples)
        true_preds = self.decide(X_batch)

         #  - torch.sum(torch.log(fake_preds + 0.0001))  #
        value = torch.sum(torch.log(true_preds + 0.0001)) + torch.sum(torch.log1p(-fake_preds + 0.0001)) # add a switch for 1 - log trick which helps to avoid saturation

        value = value / (self.batch_size * 2) # taking the mean

        return value

    def calculate_generator_score(self, z_batch):
        generated_examples = self.generator.forward(z_batch)
        fake_preds = self.decide(generated_examples)

        #value = torch.mean(torch.log(fake_preds + 0.0001)) #-torch.mean(torch.log1p(-fake_preds + 0.0001))  # add a switch for 1 - log trick which helps to avoid saturation
        value = -torch.mean(torch.log1p(-fake_preds + 0.0001))


        return value

    def train(self, X, n_updates):

        N = X.size(0)

        vals = []

        for i in range(n_updates):

            if not i % 1000:
                print("Update number {}".format(i))


            inds = np.random.randint(N, size=self.batch_size)
            inds = torch.LongTensor(inds)

            z_batch = Variable(torch.randn(self.batch_size, self.latent_dim))
            X_batch = Variable(X[inds])

            discriminator_loss = -self.calculate_game_value(z_batch, X_batch)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad()

            vals.append(-discriminator_loss.data.numpy())

            z_batch = Variable(torch.randn(self.batch_size, self.latent_dim))
            generator_loss = -self.calculate_generator_score(z_batch)

            generator_loss.backward()
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

        return vals

if __name__ == "__main__":

    data = torch.rand((1000, 1))
    latent_dim = 1

    generator = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, data.shape[1]),
    )

    discriminator = torch.nn.Sequential(
        torch.nn.Linear(data.shape[1], 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
        torch.nn.Sigmoid()
    )

    optim_generator = torch.optim.Adam
    optim_discriminator = torch.optim.Adam

    gan = GAN(generator, discriminator, optim_generator, optim_discriminator)


    for i in range(500):
        vals = gan.train(data, 1)

        plt.figure()
        xs = Variable(torch.FloatTensor(np.linspace(-10, 10, 10000).reshape(10000, 1)))
        preds = gan.decide(xs)
        plt.plot(xs.data.numpy(), preds.data.numpy())

        maps_gen = gan.generator.forward(xs)
        plt.plot(xs.data.numpy(), maps_gen.data.numpy())

        plt.savefig("./figs/{}".format(i))
        plt.close()


    plt.figure()
    plt.hist(gan.generate(100000).data.numpy())
    plt.show()

    plt.figure("Game Values")
    plt.plot(vals)
    plt.show()

    gan.decide(gan.generate(100))
    gan.decide(data)