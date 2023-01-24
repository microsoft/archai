import random
import ray

def sample_from_parent_pool(parentpool):
    if len(parentpool):
        # Finds a model by computing the lower convex hull of the 
        # genotype-accuracy pool and sampling along a tolerance region around it. 
        # Here we are just going to simulate it.
        index = random.randint(0, len(parentpool) - 1)
        genotype, accuracy = list(parentpool)[index]
        return genotype
    else:
        None


@ray.remote
def train_a_model(all_info_for_train_model):
    # Simulate training a model
    # time.sleep(random.randint(0, 3))
    up_lim = random.randint(1000, 1000000)
    counter = 0
    for i in range(up_lim):
        counter = counter + 1
    # Return trained model        
    acc = random.random()
    print(f'Trained {all_info_for_train_model}')
    return all_info_for_train_model, acc


if __name__ == "__main__":
    
    ray.init()
    num_cpus = ray.nodes()[0]['Resources']['CPU']

    # Seed genotype
    seed_genotype = 'randomstring'
    seed_genotype_accuracy = 0.1

    # Need parent, init list
    parent_set = {(seed_genotype, seed_genotype_accuracy)}

    # Sample a model from parent pool and add to init queue 
    # according to pareto frontier stuff
    model = sample_from_parent_pool(parent_set)
    model = model + '_init'
    
    # Parallel train 
    result_all_ids = [train_a_model.remote(model)]

    while len(result_all_ids):
        print(f'Num jobs {len(result_all_ids)}')
        done_id, result_all_ids = ray.wait(result_all_ids)
        print(f'After ray.wait {len(result_all_ids)}')
        # NOTE: Why do we need to index into done_id?
        trained_model, acc = ray.get(done_id[0])
        if trained_model[-4:] == 'init':
            # Augmented model just finished training
            # Start another remote job to train it
            trained_model = trained_model + '_child'
            this_id = train_a_model.remote(trained_model)
            result_all_ids.append(this_id)
        elif trained_model[-5:] == 'child':
            # Final model just finished training
            # Add it to the parent set
            # And sample another model
            parent_set.add((trained_model, acc))
            print(f'Parent set size {len(parent_set)}')
            model = sample_from_parent_pool(parent_set)
            model = model + '_init'
            this_id = train_a_model.remote(model)
            result_all_ids.append(this_id)
        
        # If there are less models being trained in parallel
        # than there are cpus (later gpus) sample more models
        # from parent pool to not waste compute
        
        if len(result_all_ids) < num_cpus:
            num_empty_res = int(num_cpus - len(result_all_ids))
            model_list = []
            for i in range(num_empty_res):
                model = sample_from_parent_pool(parent_set)
                model = model + '_init'
                model_list.append(model)
            result_ids = [train_a_model.remote(model) for model in model_list]
            result_all_ids += result_ids