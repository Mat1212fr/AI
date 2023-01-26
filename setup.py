import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

batch_size = 256

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
data = datasets.MNIST(root='road', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)



# Définir le générateur
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(100, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, 784)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        return x

# Définir le discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x

# Instancier les modèles
generator = Generator()
discriminator = Discriminator()

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)


        # Boucle d'entraînement
for epoch in range(50):
    for i, (real_images, _) in enumerate(tqdm(data_loader, desc='Training', postfix={'Epoque': epoch}, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        # Entraîner le discriminateur
        d_optimizer.zero_grad()

        # Prédire la réalité des images réelles
        real_images = real_images.view(-1, 784)
        d_real = discriminator(real_images)
        real_loss = criterion(d_real, torch.ones(d_real.size(0), 1))
        real_loss.backward()

        # Générer des images factices
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        d_fake = discriminator(fake_images)
        fake_loss = criterion(d_fake, torch.zeros(d_fake.size(0), 1))
        fake_loss.backward()

        d_loss = real_loss + fake_loss
        d_optimizer.step()

        # Entraîner le générateur
        g_optimizer.zero_grad()

        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        d_fake = discriminator(fake_images)
        g_loss = criterion(d_fake, torch.ones(d_fake.size(0), 1))
        g_loss.backward()

        g_optimizer.step()
    print("epoch: %d, d_loss: %.4f, g_loss: %.4f" % (epoch+1, d_loss.item(), g_loss.item()))
    tqdm.write(f'epoch: {epoch + 1}')
