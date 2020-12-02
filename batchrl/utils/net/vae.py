import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 latent_dim, 
                 max_action,
                 hidden_size=750):
        super(VAE, self).__init__()
        
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(state.device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: 
            return a
        return self.max_action * torch.tanh(a)


class VAEModule(object):
    def __init__(self, *args, vae_lr=1e-4, **kwargs):
        self.vae = VAE(*args, **kwargs).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def train(self, dataset, folder_name, batch_size=100, iterations=500000):
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': []}
        for i in range(iterations):
            vae_loss, recon_loss, KL_loss = self.train_step(dataset, batch_size)
            logs['vae_loss'].append(vae_loss)
            logs['recon_loss'].append(recon_loss)
            logs['kl_loss'].append(KL_loss)
            if (i + 1) % 50000 == 0:
                print('Itr ' + str(i+1) + ' Training loss:' + '{:.4}'.format(vae_loss))
                self.save('model_' + str(i+1), folder_name)
                pickle.dump(logs, open(folder_name + "/vae_logs.p", "wb"))

        return logs

    def loss(self, state, action):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        return vae_loss, recon_loss, KL_loss

    def train_step(self, dataset, batch_size=100):
        dataset_size = len(dataset['observations'])
        ind = np.random.randint(0, dataset_size, size=batch_size)
        state = dataset['observations'][ind]
        action = dataset['actions'][ind]
        vae_loss, recon_loss, KL_loss = self.loss(state, action)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return vae_loss.cpu().data.numpy(), recon_loss.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=device))