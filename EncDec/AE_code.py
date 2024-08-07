import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.utils as vutils


class AlteredMNIST(Dataset):
    def __init__(self):
        self.clean_folder = "./Data/clean"
        self.aug_folder = "./Data/aug"
        self.mapping = self.create_mapping()

    def __len__(self):
        return len(self.mapping)

    def extract_label(self, filename):
        return int(filename.split("_")[-1].split(".")[0])

    def __getitem__(self, idx):
        clean_image_path, aug_image_path = self.mapping[idx]
        aug_image = io.read_image(aug_image_path)
        clean_image = io.read_image(clean_image_path)

        aug_image = transforms.functional.rgb_to_grayscale(aug_image)
        clean_image = transforms.functional.rgb_to_grayscale(clean_image)

        aug_image = transforms.functional.to_pil_image(aug_image)
        clean_image = transforms.functional.to_pil_image(clean_image)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        aug_image = transform(aug_image)
        clean_image = transform(clean_image)

        label = self.extract_label(aug_image_path)
        return clean_image, aug_image, label


    def create_label_dictionary(self, folder):
        label_dict = {}

        for image_file in os.listdir(folder):
            label = int(image_file.split("_")[-1].split(".")[0])
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(image_file)

        return label_dict


    def create_mapping(self):
        label_dict_clean = self.create_label_dictionary(self.clean_folder)
        label_dict_aug = self.create_label_dictionary(self.aug_folder)

        mapping = []

        for label in label_dict_clean.keys():
            clean_images = label_dict_clean[label]
            aug_images = label_dict_aug[label]

            pca = PCA(0.95)
            clean_data = [transforms.functional.rgb_to_grayscale(torch.from_numpy(io.read_image(os.path.join(self.clean_folder, clean_image)).numpy())).flatten() for clean_image in clean_images]
            clean_data = np.array(clean_data)
            pca.fit(clean_data)

            gmm = GaussianMixture(n_components = 40)
            gmm.fit(pca.transform(clean_data))

            unique_clusters = set()

            clean_data_pca = pca.transform(clean_data)
            clean_cluster_predictions = gmm.predict(clean_data_pca)

            for aug_image in aug_images:
                aug_img = io.read_image(os.path.join(self.aug_folder, aug_image)).numpy()
                aug_img_pca = pca.transform(transforms.functional.rgb_to_grayscale(torch.from_numpy(aug_img)).flatten().reshape(1, -1))
                label_pred = gmm.predict(aug_img_pca)

                clean_images_cluster_indices = np.where(clean_cluster_predictions == label_pred)[0]

                distances = [np.linalg.norm(aug_img_pca - clean_data_pca[i]) for i in clean_images_cluster_indices]
                closest_clean_image_index = clean_images_cluster_indices[np.argmin(distances)]
                closest_clean_image = clean_images[closest_clean_image_index]

                mapping.append((os.path.join(self.clean_folder, closest_clean_image), os.path.join(self.aug_folder, aug_image)))
                unique_clusters.add(label_pred[0])

        random.shuffle(mapping)
        return mapping


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(encoder_block,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if out.shape != identity.shape:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 1, stride = 2),nn.BatchNorm2d(64))
        down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 1, stride = 2),nn.BatchNorm2d(128))
        down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 1, stride = 2),nn.BatchNorm2d(256))
        down4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 1, stride = 2),nn.BatchNorm2d(512))
        self.layer1 = encoder_block(in_channels = 1, out_channels = 64, stride = 2, downsample = down1)
        self.layer2 = encoder_block(in_channels = 64, out_channels = 128, stride = 2, downsample = down2)
        self.layer3 = encoder_block(in_channels = 128, out_channels = 256, stride = 2, downsample = down3)
        self.layer4 = encoder_block(in_channels = 256, out_channels = 512, stride = 2, downsample = down4)

        self.flatten = nn.Flatten()
        self.mean_vae = nn.Linear(2048, 32)
        self.logvar_vae = nn.Linear(2048, 32)
        self.fc_vae = nn.Linear(32, 2048)

        self.mean_cvae = nn.Linear(2048, 32)
        self.logvar_cvae = nn.Linear(2048, 32)
        self.fc_cvae = nn.Linear(32, 2048)
        self.label_projector = nn.Sequential(
            nn.Linear(10,32),
            nn.ReLU(),
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label


    def forward(self, x, flag, y = None):
        mean = None
        logvar = None

        if flag == 0:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            out = self.layer4(x)

        elif flag == 1:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            enc = self.flatten(x)
            mean = self.mean_vae(enc)
            logvar = self.logvar_vae(enc)
            z = self.reparameterization(mean, logvar)
            p = self.fc_vae(z)
            out = p.view(-1,512,2,2)

        elif flag == 2:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.flatten(x)
            mean = self.mean_cvae(x)
            logvar = self.logvar_cvae(x)
            z = self.reparameterization(mean, logvar)
            y_onehot = F.one_hot(y, num_classes = 10)
            z = self.condition_on_label(z,y_onehot)
            p = self.fc_cvae(z)
            out = p.view(-1,512,2,2)

        return out, mean, logvar


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, upsample = None):
        super(decoder_block, self).__init__()

        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.ConvTranspose2d(out_channels, in_channels, kernel_size = 3, stride = stride, padding = 1, output_padding = 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)

        if out.shape != identity.shape:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        up4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size = 1, stride = 2, output_padding = 1),nn.BatchNorm2d(256))
        up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size = 1, stride = 2, output_padding = 1),nn.BatchNorm2d(128))
        up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size = 1, stride = 2, output_padding = 1),nn.BatchNorm2d(64))
        up1 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size = 1, stride = 2, output_padding = 1),nn.BatchNorm2d(64))

        self.layer4 = decoder_block(in_channels = 256, out_channels = 512, stride = 2, upsample = up4)
        self.layer3 = decoder_block(in_channels = 128, out_channels = 256, stride = 2, upsample = up3)
        self.layer2 = decoder_block(in_channels = 64, out_channels = 128, stride = 2, upsample = up2)
        self.layer1 = decoder_block(in_channels = 64, out_channels = 64, stride = 2, upsample = up1)
        self.conv = nn.Conv2d(64, 1, kernel_size = 5, stride = 1, padding = 0)


    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv(x)
        return x


class AELossFn(nn.Module):
    def __init__(self):
        super(AELossFn, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, output, target):
        loss = self.loss_fn(output, target)
        return loss


class AETrainer:

    def __init__(self, data_loader, encoder, decoder, loss_fn, optimizer, device):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda") if device == "T" else torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.num_epochs = 50
        self.plot_interval = 10
        self.best_ssim = float('-inf')
        self.best_model_state = None
        self.train()

    def plot_tsne_embeddings(self, filename=None):
        embeddings = []
        labels = []

        self.encoder.eval()
        with torch.no_grad():
            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):
                clean = clean.to(self.device)
                aug = aug.to(self.device)
                logits, mean, logvar = self.encoder(aug, 0)
                embeddings.append(logits.view(logits.size(0), -1).cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        num_samples = 12000
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]


        tsne = TSNE(n_components=3, random_state=42, n_iter=700, perplexity=20)
        tsne_results = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.get_cmap('tab10')
        sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels_subset, cmap=cmap)
        cbar = plt.colorbar(sc)
        cbar.set_label('Labels')

        ax.set_xlabel('TSNE 1')
        ax.set_ylabel('TSNE 2')
        ax.set_zlabel('TSNE 3')
        ax.set_title('3D TSNE Plot')

        plt.show()
        plt.savefig(filename)
        plt.close()


    def train(self):
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_batches = 0
            train_loss = 0.0
            ssim_scores = []

            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):
                aug = aug.to(self.device)
                clean = clean.to(self.device)

                self.optimizer.zero_grad()

                logits, mean, logvar = self.encoder(aug, 0)
                outputs = self.decoder(logits)
                loss = self.loss_fn(outputs, clean)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    similarity = sum(ssim(clean[i].detach().cpu().squeeze().numpy(), outputs[i].detach().cpu().squeeze().numpy(), data_range=1.0) for i in range(clean.size(0))) / clean.size(0)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch + 1, batch_idx + 1, loss.item(), similarity))

                for i in range(clean.size(0)):
                    true_img = clean[i].detach().cpu().squeeze().numpy()
                    gen_img = outputs[i].detach().cpu().squeeze().numpy()
                    ssim_score = ssim(true_img, gen_img, data_range=1.0)
                    ssim_scores.append(ssim_score)

            train_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0

            if train_ssim > self.best_ssim:
                self.best_ssim = train_ssim
                self.best_model_state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }

            train_loss /= total_batches
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch + 1, train_loss, train_ssim))

            if (epoch + 1) % self.plot_interval == 0:
                name = f"AE_epoch_{epoch+1}"
                self.plot_tsne_embeddings(name)

        if self.best_model_state:
            torch.save(self.best_model_state, "best_model_AE.pth")

    

class VAELossFn(nn.Module):
    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, x_hat, x, mean, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), axis = 1) 
        bs = kl_div.size(0)
        kl_div = kl_div.mean()

        pw = F.mse_loss(x_hat, x, reduction = 'none')
        pw = (pw.view(bs, -1).sum(axis = 1)).mean()
        return pw + kl_div * 0.0001
    

def ParameterSelector(E, D):
    return [{'params': E.parameters()}, {'params': D.parameters()}]



class VAETrainer:
    def __init__(self, data_loader, encoder, decoder, loss_fn, optimizer, device):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda") if device == "T" else torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.num_epochs = 50
        self.plot_interval = 10
        self.best_ssim = float('-inf')
        self.best_model_state = None
        self.train()


    def plot_tsne_embeddings(self, filename=None):
        embeddings = []
        labels = []

        self.encoder.eval()
        with torch.no_grad():
            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):
                clean = clean.to(self.device)
                aug = aug.to(self.device)
                logits, mean, logvar = self.encoder(aug, 1)
                embeddings.append(logits.view(logits.size(0), -1).cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        num_samples = 12000
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]


        tsne = TSNE(n_components=3, random_state=42, n_iter=700, perplexity=20)
        tsne_results = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.get_cmap('tab10')
        sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels_subset, cmap=cmap)
        cbar = plt.colorbar(sc)
        cbar.set_label('Labels')

        ax.set_xlabel('TSNE 1')
        ax.set_ylabel('TSNE 2')
        ax.set_zlabel('TSNE 3')
        ax.set_title('3D TSNE Plot')

        plt.show()
        plt.savefig(filename)
        plt.close()



    def train(self):
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_batches = 0
            train_loss = 0.0
            ssim_scores = []

            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):

                aug = aug.to(self.device)
                clean = clean.to(self.device)
                
                self.optimizer.zero_grad()

                encoded, mean, logvar = self.encoder(aug, 1)
                x_hat = self.decoder(encoded)
                loss = self.loss_fn(x_hat, clean, mean, logvar)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    similarity = sum(ssim(clean[i].detach().cpu().squeeze().numpy(), x_hat[i].detach().cpu().squeeze().numpy(), data_range=1.0) for i in range(clean.size(0))) / clean.size(0)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch + 1, batch_idx + 1, loss.item(), similarity))

                for i in range(clean.size(0)):
                    true_img = clean[i].detach().cpu().squeeze().numpy()
                    gen_img = x_hat[i].detach().cpu().squeeze().numpy()
                    ssim_score = ssim(true_img, gen_img, data_range=1.0)
                    ssim_scores.append(ssim_score)

            train_loss /= total_batches
            train_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0

            if train_ssim > self.best_ssim:
                self.best_ssim = train_ssim
                self.best_model_state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch + 1, train_loss, train_ssim))

            if (epoch + 1) % self.plot_interval == 0:
                name = f"VAE_epoch_{epoch+1}"
                self.plot_tsne_embeddings(name)

        if self.best_model_state:
            torch.save(self.best_model_state, "best_model_VAE.pth")


class CVAELossFn(nn.Module):
    def __init__(self):
        super(CVAELossFn, self).__init__()

    def forward(self, x_hat, x, mean, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), axis=1) 
        bs = kl_div.size(0)
        kl_div = kl_div.mean()

        pw = F.mse_loss(x_hat, x, reduction='none')
        pw = (pw.view(bs, -1).sum(axis=1)).mean()
        return pw + kl_div * 0.0001



class CVAETrainer:
    def __init__(self, data_loader, encoder, decoder, loss_fn, optimizer):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.num_epochs = 50
        self.plot_interval = 10
        self.best_ssim = float('-inf')
        self.best_model_state = None
        self.train()


    def plot_tsne_embeddings(self, filename=None):
        embeddings = []
        labels = []

        self.encoder.eval()
        with torch.no_grad():
            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):
                clean = clean.to(self.device)
                aug = aug.to(self.device)
                label = label.to(self.device)

                logits, mean, logvar = self.encoder(aug, 2, label)
                embeddings.append(logits.view(logits.size(0), -1).cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        num_samples = 12000
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]


        tsne = TSNE(n_components=3, random_state=42, n_iter=700, perplexity=20)
        tsne_results = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.get_cmap('tab10')
        sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels_subset, cmap=cmap)
        cbar = plt.colorbar(sc)
        cbar.set_label('Labels')

        ax.set_xlabel('TSNE 1')
        ax.set_ylabel('TSNE 2')
        ax.set_zlabel('TSNE 3')
        ax.set_title('3D TSNE Plot')

        plt.show()
        plt.savefig(filename)
        plt.close()


    def train(self):
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_batches = 0
            train_loss = 0.0
            ssim_scores = []

            for batch_idx, (clean, aug, label) in enumerate(self.data_loader):
                aug = aug.to(self.device)
                clean = clean.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                encoded, mean, logvar = self.encoder(aug, 2, label)
                x_hat = self.decoder(encoded)
                loss = self.loss_fn(x_hat, clean, mean, logvar)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    similarity = sum(ssim(clean[i].detach().cpu().squeeze().numpy(), x_hat[i].detach().cpu().squeeze().numpy(), data_range=1.0) for i in range(clean.size(0))) / clean.size(0)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch + 1, batch_idx + 1, loss.item(), similarity))

                for i in range(clean.size(0)):
                    true_img = clean[i].detach().cpu().squeeze().numpy()
                    gen_img = x_hat[i].detach().cpu().squeeze().numpy()
                    ssim_score = ssim(true_img, gen_img, data_range=1.0)
                    ssim_scores.append(ssim_score)

            train_loss /= total_batches
            train_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0

            if train_ssim > self.best_ssim:
                self.best_ssim = train_ssim
                self.best_model_state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch + 1, train_loss, train_ssim))

            if (epoch + 1) % self.plot_interval == 0:
                name = f"CVAE_epoch_{epoch+1}"
                self.plot_tsne_embeddings(name)

        if self.best_model_state:
            torch.save(self.best_model_state, "best_model_CVAE.pth")




class AE_TRAINED:
    def __init__(self, gpu):
        self.gpu = gpu
        self.device = torch.device("cuda") if self.gpu == "T" else torch.device("cpu")
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.load_from_checkpoint("best_model_AE.pth")

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def forward(self, input_image):
        with torch.no_grad():
            encoded, _ , _ = self.encoder(input_image, 0)
            reconstructed = self.decoder(encoded)
        return reconstructed

    def from_path(self, sample, original, type):

        sample_image = io.read_image(sample)
        original_image = io.read_image(original)
        sample_image = transforms.functional.rgb_to_grayscale(sample_image)
        original_image = transforms.functional.rgb_to_grayscale(original_image)
        sample_image = transforms.functional.to_pil_image(sample_image)
        original_image = transforms.functional.to_pil_image(original_image)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        sample_image = transform(sample_image)
        original_image = transform(original_image)
        sample_tensor = sample_image.unsqueeze(0).to(self.device)
        original_tensor = original_image.unsqueeze(0).to(self.device)

        reconstructed_tensor = self.forward(sample_tensor)

        reconstructed_arr = reconstructed_tensor.squeeze().detach().cpu().numpy()
        original_arr = original_tensor.squeeze().detach().cpu().numpy()

        if type == 'SSIM':
            score = ssim(reconstructed_arr, original_arr, data_range = 1.0)
        elif type == 'PSNR':
            score = psnr(reconstructed_arr, original_arr, data_range = 1.0)

        return score
    


class VAE_TRAINED:
    def __init__(self, gpu):
        self.gpu = gpu
        self.device = torch.device("cuda") if self.gpu == "T" else torch.device("cpu")
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.load_from_checkpoint("best_model_VAE.pth")

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def forward(self, input_image):
        with torch.no_grad():
            encoded, _ , _ = self.encoder(input_image, 1)
            reconstructed = self.decoder(encoded)
        return reconstructed

    def from_path(self, sample, original, type):
        sample_image = io.read_image(sample)
        original_image = io.read_image(original)
        sample_image = transforms.functional.rgb_to_grayscale(sample_image)
        original_image = transforms.functional.rgb_to_grayscale(original_image)
        sample_image = transforms.functional.to_pil_image(sample_image)
        original_image = transforms.functional.to_pil_image(original_image)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        sample_image = transform(sample_image)
        original_image = transform(original_image)
        sample_tensor = sample_image.unsqueeze(0).to(self.device)
        original_tensor = original_image.unsqueeze(0).to(self.device)

        reconstructed_tensor = self.forward(sample_tensor)

        reconstructed_arr = reconstructed_tensor.squeeze().detach().cpu().numpy()
        original_arr = original_tensor.squeeze().detach().cpu().numpy()

        if type == 'SSIM':
            score = ssim(reconstructed_arr, original_arr, data_range=1.0)
        elif type == 'PSNR':
            score = psnr(reconstructed_arr, original_arr, data_range=1.0)

        return score



class CVAE_Generator:

    def __init__(self):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.load_from_checkpoint("best_model_CVAE.pth")


    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


    def save_image(self, digit, save_path):
        label = torch.tensor(int(digit)).unsqueeze(0).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoded = torch.randn(1, 32).to(self.device)
            label = F.one_hot(label, num_classes = 10)
            encoded = self.encoder.condition_on_label(encoded, label)
            encoded = self.encoder.fc_cvae(encoded) 
            encoded = encoded.view(-1, 512, 2, 2)
            output = self.decoder(encoded)

        vutils.save_image(output, save_path, normalize = True)



def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()


def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()