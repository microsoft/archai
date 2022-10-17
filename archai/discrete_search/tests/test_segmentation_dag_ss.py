from archai.discrete_search.search_spaces.segmentation_dag.search_space import SegmentationDagSearchSpace

if __name__ == '__main__':
    ss = SegmentationDagSearchSpace(1, img_size=(160, 96))

    model = ss.random_sample()
    print(model.arch.to_hash())
    print(model.arch.graph)

    model = ss.random_sample()
    print(model.arch.to_hash())
    print(model.arch.graph)

    # Applies mutation
    new_model = ss.mutate(model)
    print(new_model.arch.to_hash())
    print(new_model.arch.graph)

    new_model = ss.mutate(model)
    print(new_model.arch.to_hash())
    print(new_model.arch.graph)

    # Test crossover
    new_model = ss.crossover([ss.random_sample(), ss.random_sample()])
    print(new_model.arch.to_hash())
    print(new_model.arch.graph)


    # Test writing and loading
    ss.save_arch(new_model, '/tmp/test.yaml')

    print(ss.load_arch('/tmp/test.yaml').arch.to_hash())
