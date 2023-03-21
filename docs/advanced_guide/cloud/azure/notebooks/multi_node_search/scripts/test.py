from archai.trainers.cv.pl_trainer import PlTrainer
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
from torch import nn
import math
from model import MyModel
import pytorch_lightning as pl
from archai.discrete_search.search_spaces.config import ArchConfig
from mnist_data_module import MNistDataModule


def test(model_path, val_data):
    # Great, now let's test if this model works as advertised.
    import onnxruntime as ort
    import numpy as np

    count = val_data.data.shape[0]
    test = np.random.choice(count, 1)[0]
    data = val_data.data[test]

    import matplotlib.pyplot as plt

    # check what the images look like.
    plt.figure(figsize=(2,2))
    plt.imshow(data, cmap='gray')
    print(f'data has shape: {data.shape}')
    plt.axis('off')
    plt.show()

    # Now run the ONNX runtime on this the validation set.
    # You can change this to `CUDAExecutionProvider` if you have a GPU and have
    # installed the CUDA runtime.
    ort_sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    for i in ort_sess.get_inputs():
        print(f'input: {i.name}, {i.shape}, {i.type}')

    print(f'Testing {count} rows')
    failed = 0
    for i in range(val_data.data.shape[0]):
        data = val_data.data[i]
        expected = int(val_data.train_labels[i])

        while len(data.shape) < 4:
            data = np.expand_dims(data, axis=0)
        outputs = ort_sess.run(None, {'input': data.astype(np.float32) / 255.0})
        result = outputs[0]
        index = np.argmax(result)
        label = val_data.classes[index]
        if expected != index:
            # print(f'### Failed: {expected} and got {label}')
            failed += 1

    rate = (count - failed) * 100 / count
    print(f"Failed {failed} out of {count} rows")
    print(f'Inference pass rate is  {rate} %.')


def main():
    logger = pl.loggers.TensorBoardLogger('logs', name='mnist')
    logger.log_hyperparams({"epochs": 1, "lr": 1e-4})
    data = MNistDataModule('dataroot')
    data.prepare_data()
    config = ArchConfig({
        "nb_layers": 5,
        "kernel_size": 5,
        "hidden_dim": 64
    })
    model = MyModel(config)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=1, logger=logger, callbacks=[TQDMProgressBar(refresh_rate=100)])
    trainer.fit(model, data)
    result = trainer.validate(model, data)
    accuracy = result[0]['accuracy']
    model.export_onnx(data.input_shape, 'model.onnx')
    test('model.onnx', data.val_data)


if __name__ == "__main__":
    main()
