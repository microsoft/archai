## Readme

This folder contains code for running models using the Qualcomm SNPE Neural Processing SDK,
including quantizing those models and running them on the Qualcomm DSP.
This folder uses http://github.com/microsoft/olive to do the actual SNPE work.

1. **Snapdragon 888 Dev Kit** - get one of these [Snapdragon 888 HDK](https://developer.qualcomm.com/hardware/snapdragon-888-hdk) boards.

1. **Download dataset**.  Get the dataset from https://github.com/microsoft/FaceSynthetics.
The best way is using `azcopy`.  You could put them in a datasets folder,
for example: `d:\datasets\FaceSynthetics`.  Then set your `INPUT_DATASET` environment
variable pointing to this folder.

1. **Install Android NDK**. You need a working version of `adb` in your PATH.
Just download the zip file from [https://developer.android.com/ndk/downloads/](https://developer.android.com/ndk/downloads/)
and unzip it then you can set your `ANDROID_NDK_ROOT` environment variable pointing to the folder containing the
unzipped bits.

1. **Check Device USB**.  Check you can run `adb shell` to connect to your Snapdragon over USB.
You may need to run `sudo usermod -aG plugdev $LOGNAME`.

1. **Install SNPE SDK on Ubuntu 18.04**.
See [SNPE Setup](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html).
See [Neural Processing SDK Download](https://developer.qualcomm.com/downloads/qualcomm-neural-processing-sdk-ai-v1600?referrer=node/34505).
You can skip the Caffe setup, but use the `requirements.txt` pip install list, the one posted in the Qualcomm setup page
has conflicting versions.  Then set your `SNPE_ROOT` environment variable pointing to the folder containing the unzipped
bits.  If you plan to use Qualcomm hardware devices then set the `SNPE_ANDROID_ROOT` to the same place as `SNPE_ROOT`.

1. **Install Archai**.  In your Python 3.8 Conda environment run:

    ```
    git clone https://github.com/microsoft/archai.git
    cd archai
    pip install -e .[dev]
    ```

1. **Install required packages including Olive **

    ```
    pushd tasks/face_segmentation/aml
    pip install -r requirements.txt
    ```

1. Let Olive configure SNPE
    ```
	python -m olive.snpe.configure
    ```

    **If you run into a protobuf inconsistency with Python 3.8 you can workaround
    it by setting the folloiwng env. variable:**
    ```
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    ```

1. **Create experiment folder**.  The subsequent scripts all assume you are in a folder for running your experiment.
    ```
    mkdir ~/experiment1
    cd ~/experiment1
    ```

    In this folder we will build the following files:
    - data/test - the test image set for the device
    - data/quant - the image dataset for quantizing the model
    - snpe_models/model.quant.dlc - the quantized model

1. **Prepare data**. Run `python create_data.py --help`, this scripts creates data for both
   quantization and test and puts it in  your local experiment folder under `data/test` and
   `data/quant`.  For example:

    ```
    python create_data.py --input ~/datasets/FaceSynthetics --count 1000 --dim 256
    ```

1. **Convert and quantize model**. You can use `test_snpe.py` to convert a .onnx model to .dlc and
quantize it.  For example:
    ```
    python test_snpe.py --quantize --model model.onnx
    ```
This can take about 10 minutes depending on the size of your quantization data set and the size of
your model.

1. **Run test images on device**. You can use `test_snpe.py` to test your quantized model on a
Qualcomm 888 dev board. You can find the device id using `adb devices`:
    ```
    python test_snpe.py --device e6dc0375 --images ./data/test --model model.onnx --dlc ./snpe_models/model.quant.dlc
    ```

6. **Performance benchmark SNPE model**.
    ```
    python test_snpe.py --device e6dc0375 --benchmark --images ./data/test --model model.onnx  --dlc ./snpe_models/model.quant.dlc
    ```
