import json

from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.utils.augmented_trainer import train_and_eval
from archai.common.common import common_init, expdir_abspath

logger = get_global_logger()


if __name__ == "__main__":
    conf = common_init(
        config_filepath="confs/aug/aug_train_cifar.yaml",
        param_args=["--autoaug.loader.aug", "fa_reduced_cifar10", "--common.experiment_name", "autoaug_train"],
    )
    import time

    t = time.time()
    save_path = expdir_abspath("model.pth")

    # result = train_and_eval(conf, val_ratio=conf['val_ratio'], val_fold=conf['val_fold'],
    #                         save_path=save_path, only_eval=conf['only_eval'], metric='test')

    # TODO: Will fail if val_ratio=0 since we are not using latest training infrastructure
    # TODO: Move val_ratio, val_fold, metric to config file
    result = train_and_eval(conf, val_ratio=0.2, val_fold=0, save_path=save_path, only_eval=False, metric="test")
    elapsed = time.time() - t

    logger.info("training done.")
    logger.info("model: %s" % conf["autoaug"]["model"])
    logger.info("augmentation: %s" % conf["autoaug"]["loader"]["aug"])
    logger.info("\n" + json.dumps(result, indent=4))
    logger.info("elapsed time: %.3f Hours" % (elapsed / 3600.0))
    logger.info("top1 error in testset: %.4f" % (1.0 - result["top1_test"]))
    logger.info("Save path: %s" % save_path)
