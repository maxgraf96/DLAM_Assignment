import torch
from torch.nn import functional as F
from torchvision.utils import save_image


class Model:
    def __init__(self, model, device, batch_size, log_interval):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.log_interval = log_interval

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x[0][0], reduction='mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Normalise KLD by batch size and size of spectrogram
        KLD /= self.batch_size

        return BCE + KLD

    def train(self, epoch, train_loader, optimizer):
        # Set current epoch in model (used for plotting)
        self.model.set_epoch(epoch)
        self.model.train()
        train_loss = 0
        for (batch_idx, data) in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            # Use if nans start showing up again
            # nans = (recon_batch != recon_batch).any()
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    # comparison = torch.cat([data[:n],
                    #                         recon_batch.view(self.batch_size, 1, SPEC_DIMS_W, SPEC_DIMS_H)[:n]])
                    # save_image(comparison.cpu(),
                    #            'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def generate(self, spec):
        with torch.no_grad():
            sample = torch.from_numpy(spec).to(self.device)
            sample = self.model.forward(sample.view(1, 1, spec.shape[1], spec.shape[0]))

            return sample[0]

    # def save_sample_ANN(self):
    #     with torch.no_grad():
    #         # 3 => 3x1 matrix, 4 => number of channels in the bottleneck dimension of VAE /, spec height, spec width
    #         sample = torch.randn(64, ZDIMS_ANN).to(device)
    #         sample = self.model.decode(sample).cpu()
    #         save_image(sample.view(64, 1, SPEC_DIMS_W, SPEC_DIMS_H),
    #                    'results/sample_' + str(epoch) + '.png')
