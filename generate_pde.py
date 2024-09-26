import yaml
from argparse import ArgumentParser
from scripts import generate_burgers, generate_darcy, generate_poisson, generate_helmholtz, generate_ns_nonbounded, generate_ns_bounded

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    options = parser.parse_args()
    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']
    if name == 'Burgers':
        print('Solving Burgers equation...')
        generate_burgers(config)
    elif name == 'Darcy':
        print('Solving Darcy Flow equation...')
        generate_darcy(config)
    elif name == 'Poisson':
        print('Solving Poisson equation...')
        generate_poisson(config)
    elif name == 'Helmholtz':
        print('Solving Helmholtz equation...')
        generate_helmholtz(config)
    elif name == 'NS-NonBounded':
        print('Solving non-bounded NS equation...')
        generate_ns_nonbounded(config)
    elif name == 'NS-Bounded':
        print('Solving bounded NS equation...')
        generate_ns_bounded(config)
    else:
        print('PDE not found')
        exit(1)