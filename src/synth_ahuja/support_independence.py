import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import copy


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input layer is 1 since X is 1-dimensional
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),  # First hidden layer
            nn.LeakyReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Second hidden layer
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim) 
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
    
class PolyDecoder(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim, poly_degree, device):
        super(PolyDecoder, self).__init__()        
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.poly_degree = poly_degree
        self.device = device
        
        self.total_poly_terms = self.compute_total_polynomial_terms()
        
        self.coff_matrix= nn.Sequential(
                    nn.Linear(self.total_poly_terms, self.data_dim),  
                )        
        
    def forward(self, z):
        
        x=[]
        for idx in range(z.shape[0]):
            x.append( self.compute_decoder_polynomial(z[idx, :]))
        x= torch.cat(x, dim=0)
        x= self.coff_matrix(x)
        
        return x
    
    
    def compute_total_polynomial_terms(self):
        count=0
        for degree in range(self.poly_degree + 1):
            count+= pow(self.latent_dim, degree)
        return count

    
    def compute_kronecker_product(self, degree, latent):
        if degree ==0:
            out = torch.tensor([1]).to(self.device)        
        else:
            out = torch.clone(latent)
            for idx in range(1, degree):
                out= torch.kron(out, latent)
        return out

    def compute_decoder_polynomial(self, latent):
        out=[]
        for degree in range(self.poly_degree + 1):
    #         print('Computing polynomial term of degree ', degree)
            out.append(self.compute_kronecker_product(degree, latent))

        out= torch.cat(out)
        out= out.view((1,out.shape[0]))    
        return out

class AE_poly(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(AE_poly, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input layer is 1 since X is 1-dimensional
            nn.LeakyReLU(),
            nn.Linear(256, 128),  # Input layer is 1 since X is 1-dimensional
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),  # First hidden layer
            nn.LeakyReLU()
        )
        # Decoder
        self.decoder = PolyDecoder(input_dim, latent_dim, 2,'cuda').to('cuda')
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    

"""
train and predict
"""

def train_and_predict(X, Z, num_epochs = 100, batch_size = 64, learning_rate = 1e-3):

    def get_cross_correlation(pred_latent, true_latent, batch_size = 5000):
        
        num_samples = pred_latent.shape[0]
        dim = pred_latent.shape[1]
        total_batches = int( num_samples / batch_size )  

        mcc_arr= []
        for batch_idx in range(total_batches):
            
            z_hat = copy.deepcopy(pred_latent[ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
            z = copy.deepcopy(true_latent[ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
            batch_idx += 1
            
            cross_corr = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    cross_corr[i,j]= (np.cov( z_hat[:,i], z[:,j] )[0,1]) / ( np.std(z_hat[:,i])*np.std(z[:,j]) )


            cost = -1*np.abs(cross_corr)
            row_ind, col_ind = linear_sum_assignment(cost)
            score = 100*( -1*cost[row_ind, col_ind].sum() )/(dim)
            #print(-100*cost[row_ind, col_ind])
        
            mcc_arr.append(score)
        
        return mcc_arr

    # Initialize the autoencoder, loss function, and optimizer
    device = 'cuda'
    model = Autoencoder(X.shape[1], Z.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #build the dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = data.TensorDataset(X_tensor, X_tensor)  # Input and target are the same for autoencoders
    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Training the autoencoder
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            x_batch, _ = batch_data  # We only need the image, not the label
            # Forward pass
            output = model(x_batch.to(device))
            loss = criterion(output, x_batch.to(device))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not epoch%5:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    """
    lets check MCC
    """

    ## compute MCC
    pred_Zs = model.encoder(torch.tensor(X, dtype=torch.float32).to(device)).detach().cpu().numpy()

    ## from ahuja's code
    mcc = get_cross_correlation(pred_Zs, Z, batch_size=batch_size)

    # mcc = []
    # for i in range(true_Zs_observational.shape[1]):
    #     corr, _ = pearsonr(true_Zs_observational[:,i], pred_Zs_observational[:,i])
    #     mcc.append(corr)

    return mcc

"""
generate synth data
"""

def compute_total_polynomial_terms(poly_degree, latent_dim):
    count=0
    for degree in range(poly_degree+1):
        count+= pow(latent_dim, degree)
    return count

def compute_kronecker_product(degree, latent):
    if degree ==0:
        out= np.array([1])        
    else:
        out=copy.deepcopy(latent)
        for idx in range(1, degree):
            out= np.kron(out, latent) 
    return out

def compute_decoder_polynomial(poly_degree, latent):
    out=[]
    for degree in range(poly_degree+1):
        out.append(compute_kronecker_product(degree, latent))
        
    out = np.concatenate(out)
    out = np.reshape(out, (1,out.shape[0]))   

    return out


##define some params
poly_degree, latent_dim, seed, use_interventions = 2, 3, 42, False
data_dim = 1_000
poly_size = compute_total_polynomial_terms(poly_degree, latent_dim)
coff_matrix = np.random.multivariate_normal(np.zeros(poly_size), np.eye(poly_size), size=data_dim).T

"""
define Zs
"""

# Generate observational data
np.random.seed(seed)
Z1 = np.random.normal(3, 1, 1000)
Z2 = np.random.normal(2, 1, 1000)
Z3 = np.random.normal(-4, 1, 1000)

z = np.vstack([Z1, Z2, Z3]).T

## add interventions
#intervention_indices = [0,0,0,0,1,0,1,1,2,2,2,2,0,0,1,2,0]
intervention_indices = []
for idx, intervene_idx in np.ndenumerate(intervention_indices):
    z[idx, intervene_idx] = 2.0

""""
generate x
"""

x=[]
for idx in range(z.shape[0]):
    x.append(compute_decoder_polynomial(poly_degree, z[idx, :]))
x = np.concatenate(x, axis=0)

##
x1 = np.matmul(x[:, :1+latent_dim], coff_matrix[:1+latent_dim, :])
x2 = np.matmul(x[:, 1+latent_dim:], coff_matrix[1+latent_dim:, :])
norm_factor= 0.5 * np.max(np.abs(x2)) / np.max(np.abs(x1)) 
x2 = x2 / norm_factor

x = (x1+x2)


"""
do it for observational and interventional
"""

##only observational
mcc_observational, mcc_interventional = [], []

for seed in tqdm([42, 81, 1, 13, 2], desc= 'selecting seed'):
    
    #observational
    mcc_observational.append(train_and_predict(x, z))

    #observational + hard interventions
    mcc_interventional.append(train_and_predict(seed=seed, use_interventions=True))


mean_obs, std_obs = np.mean(mcc_observational), np.std(mcc_observational)
mean_int, std_int = np.mean(mcc_interventional), np.std(mcc_interventional)
print(f"observational:  {mean_obs} +- {std_obs}, interventional: {mean_int} +- {std_int}")