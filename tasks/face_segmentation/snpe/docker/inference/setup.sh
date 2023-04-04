SNPE_SDK_ZIP=snpe-2.5.0.4052.zip
SNPE_SDK_ROOT=snpe-2.5.0.4052
ANDROID_NDK_ZIP=android-ndk-r23b-linux.zip
ANDROID_NDK_ROOT=android-ndk-r23b

set -e

if ! [ -f ${ANDROID_NDK_ZIP} ]; then
    curl -O --location "https://dl.google.com/android/repository/${ANDROID_NDK_ZIP}"
fi

if ! [ -f ${SNPE_SDK_ZIP} ]; then
    echo "Please download the ${SNPE_SDK_ZIP} from :"
    echo "https://developer.qualcomm.com/downloads/qualcomm-neural-processing-sdk-linux-v2050"
    echo "and place the file in this folder."
    exit 1
fi

docker build . --build-arg "SNPE_SDK_ZIP=${SNPE_SDK_ZIP}" --build-arg "SNPE_SDK_ROOT=${SNPE_SDK_ROOT}" --build-arg "ANDROID_NDK_ZIP=${ANDROID_NDK_ZIP}" --build-arg "ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}"
