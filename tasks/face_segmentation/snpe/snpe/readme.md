## Readme

This folder contains code for running models using the Qualcomm SNPE Neural Processing SDK,
including quantizing those models and running them on the Qualcomm DSP.

1. **Snapdragon 888 Dev Kit** - get one of these [Snapdragon 888 HDK](https://developer.qualcomm.com/hardware/snapdragon-888-hdk) boards.

1. **Download dataset**.  Get the dataset from https://github.com/microsoft/FaceSynthetics.
The best way is using `azcopy`.  You could put them in a datasets folder,
for example: `d:\datasets\FaceSynthetics`.  Then set your `INPUT_DATASET` environment
variable pointing to this folder.

1. **Install Android NDK**. You need a working version of `adb` in your PATH.  If you are
on Windows you can use the VS 2022 installer to install the
mobile development package which includes the Android NDK
and `adb` lives in `C:\Microsoft\AndroidSDK\25\platform-tools\`.
If you are on Linux, just download the zip file from [https://developer.android.com/ndk/downloads/](https://developer.android.com/ndk/downloads/) and unzip it then you can set your `ANDROID_NDK_ROOT` environment variable pointing to the folder containing the unzipped bits.

1. **Check Device USB**.  Check you can run `adb shell` to connect to your Snapdragon
over USB. (You won't be able to do this on WSL2, but you
can do it from Windows).  So the `convert_onnx.sh` that uses
the SNPE SDK runs in Ubuntu, but everything else can run on the Windows side.
You may need to run `sudo usermod -aG plugdev $LOGNAME`.

1. **Install SNPE SDK on Ubuntu 18.04**.
See [SNPE Setup](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html).
See [Neural Processing SDK Download](https://developer.qualcomm.com/downloads/qualcomm-neural-processing-sdk-ai-v1600?referrer=node/34505).  It works in Windows WSL2 with an Anaconda Python 3.6 environment.  You can skip the
Caffe setup, but use the `requirements.txt` pip install list, the one
posted in the Qualcomm setup page has conflicting versions.  Then set your `SNPE_ROOT` environment variable
pointing to the folder containing the unzipped bits.

1. **Install python packages**.  In your Python 3.6 Conda environment run `pip install -r requirements.txt`

1. **Create experiment folder**.  The subsequent scripts all assume you are in a folder for running your
experiment.
```
    mkdir experiment1
    cd experiment1
```
In this folder we will build the following files:
- data/test - the test image set for the device
- data/quant - the image dataset for quantizing the model
- model - your original model to be converted to .dlc
- snpe_models/model.quant.dlc - the quantized model

The `azure/runner.py` does all this for you, but it's also nice to be able to do these steps manually if
you need to double check something.

1. **Prepare data**. Run `python create_data.py --help`, this scripts creates data for both quantization and test and puts it in  your local experiment folder under `data/test` and `data/quant`.
For example:
    ```
    python create_data.py --input d:\datasets\FaceSynthetics --count 100 --dim 256
    ```

1. **Copy your tensorflow model**.  In your experiment folder create a folder named `model` and copy your
trained tensorflow model into this folder.  You should have something like:
    ```
    checkpoint
    model.pb
    model_11.cptk.data-00000-of-00001
    model_11.cptk.index
    model_11.cptk.meta
    ```

1. **Copy your onnx model**.  In your experiment folder create a folder named `model` and copy your
trained ONNX model into this folder.  You should have something like:
    ```
    model.onnx
    ```

1. **Setup your snpe environment**.  For onnx toolset use the following:
    ```
    pushd ~/snpe/snpe-1.64.0.3605
    source bin/envsetup.sh -o ~/anaconda3/envs/snap/lib/python3.6/site-packages/onnx
    ```
    For tensorflow use:
    ```
    pushd ~/snpe/snpe-1.64.0.3605
    source bin/envsetup.sh -ot ~/anaconda3/envs/snap/lib/python3.6/site-packages/tensorflow
    ```

1. **Convert and quantize model**. Inside your experiment folder, run `bash convert_tf.sh` or `bash
convert_onnx.sh modelname` which uses your SNPE SDK install to convert the tensorflow model in the
model folder to a Qualcomm .dlc file, and then runs the SNPE quantization tool on that using the
`quant` dataset to produce a quantized version of that model, so the output is
`snpe_models/model.dlc` and `snpe_models/model.quant.dlc` and it is the quantized model we will run
on the dev board using the DSP processor.

1. **Run test images on device**. Inside your experiment folder, run `python run.py --help` and you
will see the args you need to pass in order to upload the test images to the device, upload the
model, then run the test on the DSP processor, then download the results. For example:
    ```
    python run.py --images --model model.quant.dlc --run --download
    ```

6. **Profile SNPE model**.
Update `benchmark/config.json` so it has the right paths and
run `cd benchmark && bash run_benchmark.sh && cd ..`
The above command line will generate a csv file with per layer profiling result.
See [Performance Analysis Using Benchmarking Tools](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/learning-resources/vision-based-ai-use-cases/performance-analysis-using-benchmarking-tools)
