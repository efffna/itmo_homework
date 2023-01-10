import torch; torch.manual_seed(0)
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import tqdm
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.reshape_dim = int(latent_dims ** (0.5))
        self.conv1x1 = nn.Conv2d(1, 256, kernel_size=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, z: torch.Tensor):
        z = z.unflatten(1, (1, self.reshape_dim, self.reshape_dim))
        z = F.relu(self.conv1x1(z))
        z = F.relu(self.up(self.conv4(z)))
        z = F.relu(self.up(self.conv3(z)))
        z = F.relu(self.up(self.conv2(z)))
        z = F.relu(self.up(self.conv1(z)))

        z = torch.sigmoid(z)
        return z

    
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)

        self.linear1 = nn.Linear(64, latent_dims)
        self.linear2 = nn.Linear(64, latent_dims)
        self.pool = nn.MaxPool2d(2)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.conv1x1(x))


        x = torch.flatten(x, start_dim=1)
        mu =  self.linear1(x)
        sigma = torch.exp(self.linear2(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def VAE_train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    pbar = tqdm.tqdm(range(epochs))
    autoencoder.train()
    for epoch in pbar:
        losses = []
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            losses.append(loss.cpu().item())
            pbar.set_description('Loss: ' + str(loss.mean().item()))
        print("Epoch loss: ", sum(losses) / len(losses))
    return autoencoder


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        z = PCA(n_components=2).fit_transform(z)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.savefig('vae_latent_space.png')
            break

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 128
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(128, 128)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1.to(device))
    z_2 = autoencoder.encoder(x_2.to(device))

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z.to(device))
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255
    images_list = [Image.fromarray(img.reshape(128, 128, 3).astype(np.uint8)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)

def main():
    latent_dims = 64
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48232,), (0.23051,))
    ])
    data = torch.utils.data.DataLoader(
            torchvision.datasets.CelebA('./data',
                transform=transform,
                target_type='identity',
                download=False),
            batch_size=256,
            shuffle=True)

    vae = VariationalAutoencoder(latent_dims).to(device) # GPU
    vae = VAE_train(vae, data, epochs=5)

    plot_latent(vae, data, 100)

    imgs = next(iter(data))[0]
    interpolate_gif(vae, 'gif', imgs[0], imgs[1])

if __name__ == '__main__':
    main()