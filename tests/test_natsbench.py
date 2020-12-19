from nats_bench import create









def main():

    # Create the API instance for the topology search space in NATS
    api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)

    # slow mode (NOTE: uses up 30GB RAM)
    # api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-tss-v1_0-3ffb9.pickle.pbz2', 'tss', fast_mode=False, verbose=True)
    
    # Query the loss / accuracy / time for n-th candidate architecture on CIFAR-10
    # info is a dict, where you can easily figure out the meaning by key
    info = api.get_more_info(100, 'cifar10')

    # Query the flops, params, latency. info is a dict.
    cost_info = api.get_cost_info(12, 'cifar10')

    # Show information of an architecture index
    api.show(100)

    data = api.query_by_index(284, dataname='cifar10', hp='12')

    print('dummy')








if __name__ == '__main__':
    main()