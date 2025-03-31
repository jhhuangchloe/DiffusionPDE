import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io

def random_index_and_cylinder(center, radius, k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k% indices from a [grid_size, grid_size] grid as well as the known boundary of the cylinder.'''
    np.random.seed(seed)
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mask[i, j] = 1
            else:
                if np.random.rand() < k:
                    mask[i, j] = 1
    num_ones = mask.sum().item()
    return mask, num_ones

def cylinder_index(center, radius, grid_size, device=torch.device('cuda')):
    '''return the known boundary of the cylinder.'''
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mask[i, j] = 1
    num_ones = mask.sum().item()
    return mask, num_ones

def get_ns_bounded_loss(a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
    """Return the loss of the bounded NS equation and the observation loss."""
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    pde_loss = grad_x_next_x + grad_x_next_y
    pde_loss = pde_loss.squeeze()
    pde_loss[0, :] = 0
    pde_loss[-1, :] = 0
    pde_loss[:, 0] = 0
    pde_loss[:, -1] = 0
    
    a_GT = a_GT.view(1, 1, 128, 128)
    u_GT = u_GT.view(1, 1, 128, 128)
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask
    
    return pde_loss, observation_loss_a, observation_loss_u
    

def generate_ns_bounded(config):
    """Generate bounded NS equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = np.load(datapath)
    a_GT = data[offset, :, :, 4]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
    u_GT = data[offset, :, :, 8]
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    c_x = config['data']['c_x']
    c_y = config['data']['c_y']
    center = (c_x, c_y) # center of the 2D cylinder
    radius = config['data']['radius'] # radius of the 2D cylinder
    
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
    problem_type = config['test']['problem_type']
    if problem_type == 'both':
        known_index_a, a_count = random_index_and_cylinder(center, radius, 0.01, 128, seed=1, device=device)
        known_index_u, u_count = random_index_and_cylinder(center, radius, 0.01, 128, seed=0, device=device)
    elif problem_type == 'forward':
        known_index_a, a_count = random_index_and_cylinder(center, radius, 0.01, 128, seed=1, device=device)
        known_index_u, u_count = cylinder_index(center, radius, 128, device=device)
    elif problem_type == 'inverse':
        known_index_a, a_count = cylinder_index(center, radius, 128, device=device)
        known_index_u, u_count = random_index_and_cylinder(center, radius, 0.01, 128, seed=0, device=device)
    
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
        a_N = x_N[:,0,:,:].unsqueeze(0)
        u_N = x_N[:,1,:,:].unsqueeze(0)
        a_N = (a_N*10).to(torch.float64)
        u_N = (u_N*10).to(torch.float64)
        
        # Compute the loss
        pde_loss, observation_loss_a, observation_loss_u = get_ns_bounded_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, device=device)
        L_pde = torch.norm(pde_loss, 2)/(128*128)
        L_obs_a = torch.norm(observation_loss_a, 2)/a_count
        L_obs_u = torch.norm(observation_loss_u, 2)/u_count
        grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
        zeta_obs_a = config['generate']['zeta_obs_a']
        zeta_obs_u = config['generate']['zeta_obs_u']
        zeta_pde = config['generate']['zeta_pde']
        if i <= 0.8 * num_steps:
            x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
        else:
            x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde
    
    ############################ Save the data ############################
    x_final = x_next
    a_final = x_final[:,0,:,:].unsqueeze(0)
    u_final = x_final[:,1,:,:].unsqueeze(0)
    a_final = (a_final*10).to(torch.float64)
    u_final = (u_final*10).to(torch.float64)
    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
    relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
    print(f'Relative error of a: {relative_error_a}')
    print(f'Relative error of u: {relative_error_u}')
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()
    scipy.io.savemat('ns_bounded_results.mat', {'a': a_final, 'u': u_final})
    print('Done.')