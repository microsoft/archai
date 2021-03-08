# this is installed via pip
from nats_bench import create

from archai.algos.natsbench.lib.models import get_cell_based_tiny_net

def main():

    # Create the API instance for the topology search space in NATS
    api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=True)

    # slow mode (NOTE: uses up lots of RAM)
    # api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-tss-v1_0-3ffb9.pickle.pbz2', 'tss', fast_mode=False, verbose=True)
    
    # Query the loss / accuracy / time for n-th candidate architecture on CIFAR-10
    # info is a dict, where you can easily figure out the meaning by key
    info = api.get_more_info(100, 'cifar10')

    # Query the flops, params, latency. info is a dict.
    cost_info = api.get_cost_info(12, 'cifar10')

    # Show information of an architecture index
    # api.show(100)

    # Query by index to get all runs individually (see paper appendix)
    data = api.query_by_index(284, dataname='cifar10', hp='90')
    data[777].train_acc1es[89]

    info = api.get_more_info(1528, 'cifar10', hp=90, is_random=False)

    # Create the instance of th 12-th candidate for CIFAR-10
    config = api.get_net_config(12, 'cifar10')
    # network is a nn.Module subclass. the last few modules have names
    # lastact, lastact.0, lastact.1, global_pooling, classifier 
    # which we can freeze train as usual
    network = get_cell_based_tiny_net(config)

    print('Done')



if __name__ == '__main__':
    main()