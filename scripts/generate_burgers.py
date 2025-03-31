import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io

def random_sensor(k, grid_size, seed=0, device=torch.device('cuda')):
    """Return a index list with k sensors randomly placed in a grid of size [grid_size, grid_size]."""
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    known_index = torch.randperm(grid_size, device=device)[:k]
    for i in known_index:
        index[:, i]=1
    return index

def get_burger_loss(u, u_GT, mask, device=torch.device('cuda')):
    """Return the loss of the Burgers' equation and the observation loss."""
    u = u.view(1, 1, 128, 128)
    u_GT = u_GT.view(1, 1, 128, 128)
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2 
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2 
    u_t = F.conv2d(u, deriv_t, padding=(1, 0)) 
    u_x = F.conv2d(u, deriv_x, padding=(0, 1)) 
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))

    pde_loss = u_t + u * u_x - 0.01 * u_xx
    pde_loss = pde_loss.squeeze()
    observation_loss = u - u_GT
    observation_loss = observation_loss.squeeze()
    observation_loss = observation_loss * mask
    return pde_loss, observation_loss

def generate_burgers(config):
    """Generate Burgers' equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    init_state = data['input']
    init_state = torch.tensor(init_state, dtype=torch.float64, device=device)
    ground_truth = data['output'][offset, :, :]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float64, device=device)
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    
    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    selected_index = random_sensor(5, 128)
    
    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Scale the data back
        x_N = (x_N * 1.415).to(torch.float64)
        
        # Compute the loss
        pde_loss, observation_loss = get_burger_loss(x_N, ground_truth, selected_index, device)
        L_pde = torch.norm(pde_loss, 2)/(128*128)
        L_obs = torch.norm(observation_loss, 2)/(128*5)
        grad_x_cur_obs = torch.autograd.grad(outputs=L_obs, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
        zeta_obs = config['generate']['zeta_obs']
        zeta_pde = config['generate']['zeta_pde']
        if i <= 0.8 * num_steps:
            x_next = x_next - zeta_obs * grad_x_cur_obs
        else:
            x_next = x_next - zeta_obs / 10 * grad_x_cur_obs - zeta_pde * grad_x_cur_pde
    
    ############################ Save the data ############################
    x_final = (x_next * 1.415).to(torch.float64)
    relative_error = torch.norm(x_final - ground_truth, 2)/torch.norm(ground_truth, 2)
    print(f'Relative error: {relative_error}')
    x_final = x_final.to('cpu').detach().numpy()
    np.save(f'burger-results.npy', x_final)
    print('Done.')