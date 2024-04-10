import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import src.utils as ut
import importlib
import math
importlib.reload(ut)

class SparseGO(torch.nn.Module):

    def __init__(self, input_genes, latent_go, relation_dict, bias = True, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_genes = input_genes
        self.latent_go = latent_go
        self.relation_dict = relation_dict

        ## create sparse weight matrix according to GO relationships
        mask = torch.zeros((self.input_genes, self.latent_go), **factory_kwargs)
        for i in range(self.input_genes):
            for latent_go in self.relation_dict[i]:
                mask[i, latent_go] = 1

        self.register_buffer('mask', mask)
        self.mask = mask
        self.weight = nn.Parameter(torch.empty((self.input_genes, self.latent_go), **factory_kwargs))
        self.bias = None

        # if bias:
        #     self.bias = nn.Parameter(torch.empty(latent_go, **factory_kwargs))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        return (x @ (self.weight * self.mask))
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, reldict):
        super(Autoencoder, self).__init__()

        # Encoder
        #self.encoder = SparseGO(input_size, hidden_size, reldict)
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        #self.lrelu = nn.LeakyReLU(True)

        # Decoder
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)

    def forward(self, x):

        z = self.encoder(x)
        #h = self.lrelu(z)
        x = self.decoder(z)
        # x = self.decoder2(h)

        return x
    
# Example usage
N = 1000  # Number of samples
X, latent_data = ut.generate_deterministic_synth_data(N)
print(f"mean of latent features: {latent_data.mean(axis=0)}")

# Model, Loss Function, and Optimizer
relation_dict = {0:[0], 1: [0]}

model = Autoencoder(input_size = X.shape[1], hidden_size = latent_data.shape[1], reldict = relation_dict).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(torch.tensor(X).float(), batch_size=32, shuffle=True)

# Training Loop
epochs = 100
for epoch in range(epochs):

    train_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.cuda())
        loss = criterion(output, data.cuda())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    if not epoch%5:
        z = model.encoder(data.cuda()).detach().cpu().numpy()
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}, encoder weight: {model.encoder.weight}, decoder weight: {model.decoder.weight}')
        
        
