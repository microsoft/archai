# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import cv2      # pip install opencv-contrib-python
import numpy as np
import os
import glob
import onnxruntime as rt


class ImageStream:
    def __init__(self):
        self.new_frame = False
        self.frame = None

    def load_next_image(self):
        """ advance to next image in the stream """
        return None

    def get_next_frame(self):
        """ return the current image """
        return np.copy(self.frame)


class VideoStream(ImageStream):
    def __init__(self, camera):
        super(VideoStream, self).__init__()
        self.camera = camera
        self.capture_device = cv2.VideoCapture(self.camera)
        self.load_next_image()

    def get_next_frame(self):
        ret, self.frame = self.capture_device.read()
        if (not ret):
            raise Exception('your capture device is not returning images')
        return self.frame

    def load_next_image(self):
        # the video stream is live, no need to "advance" to the next frame
        pass


class FileImageStream(ImageStream):
    def __init__(self, image_file_or_folder):
        super(FileImageStream, self).__init__()
        if os.path.isdir(image_file_or_folder):
            image_file_or_folder = os.path.join(image_file_or_folder, '*')

        self.images = glob.glob(image_file_or_folder)
        self.image_pos = 0
        self.image_filename = None
        self.load_next_image()

    def load_next_image(self):
        frame = None
        while frame is None and self.image_pos < len(self.images):
            filename = self.images[self.image_pos]
            self.image_filename = filename
            frame = cv2.imread(filename)
            if frame is None:
                print("Error loading image: {}".format(filename))
            else:
                self.new_frame = True
                self.frame = frame
            self.image_pos += 1
        return frame


class InferenceDisplay:
    def __init__(self, source: ImageStream):
        self.source = source

    def run(self, model):
        onnx_session = rt.InferenceSession(model)
        inputs = onnx_session.get_inputs()
        if len(inputs) > 1:
            raise Exception("This script only supports models with a single input")

        input_shape = inputs[0].shape  # should be [1,3,256,256]
        image_size = tuple(input_shape[-2:])
        input_name = inputs[0].name

        while True:
            input_image = self.source.get_next_frame()
            if input_image is None:
                break
            input_image = self.resize_image(input_image, image_size)
            # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            # onnx expects the rgb channel to come first
            frame = input_image.transpose(2, 0, 1)
            # onnx expects the batch size to be the first dimension
            frame = np.expand_dims(frame, axis=0)
            # onnx model expects float32, and scaled down to the range [0,1.0].
            frame = frame.astype(np.float32) / 255
            results = onnx_session.run(None, input_feed={input_name: frame})
            frame = self.color_map_result(results[0])

            # create side by side input image and result.
            new_image = np.hstack((input_image, frame))
            cv2.imshow('frame', new_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                self.source.load_next_image()
            elif key == ord('q') or key == ord('x') or key == 27:
                break

    def color_map_result(self, result : np.array):
        # The result should be type: float32 (1,18,255,255) so we have to transpose that to (255,255,18)
        # and also need to do an argmax while we are at it to find the "predicted category".
        # and we also need to normalize it to the range 0-255 which we can also do in one pass.
        # so we can do all of that in one line of code:
        num_classes = result.shape[1]
        result = result[0].transpose(1, 2, 0)
        predictions = np.argmax(result, axis=2) * (255.0 / num_classes)
        return cv2.applyColorMap(predictions.astype(np.uint8), cv2.COLORMAP_JET)

    def resize_image(self, image, newSize):
        """Center crop and resizes image to newSize. Returns image as numpy array."""
        if image.shape[0] > image.shape[1]:  # Tall (more rows than cols)
            rowStart = int((image.shape[0] - image.shape[1]) / 2)
            rowEnd = rowStart + image.shape[1]
            colStart = 0
            colEnd = image.shape[1]
        else:  # Wide (more cols than rows)
            rowStart = 0
            rowEnd = image.shape[0]
            colStart = int((image.shape[1] - image.shape[0]) / 2)
            colEnd = colStart + image.shape[0]

        cropped = image[rowStart:rowEnd, colStart:colEnd]
        resized = cv2.resize(cropped, newSize, interpolation=cv2.INTER_LINEAR)
        return resized


def main():
    arg_parser = argparse.ArgumentParser(
        "This script will run inference on a given .onnx model using the onnx runtime\n" +
        "and show the results of that inference on a given camera or static image input." +
        "Press q, x, or ESC to exit. Press SPACE to advance to the next image."
    )
    arg_parser.add_argument("--camera", type=int, help="the camera id of the webcam", default=0)
    arg_parser.add_argument("--images", help="path to image files (supports glob wild cards and folder names).")
    arg_parser.add_argument("--model", help="path to .onnx model.")
    args = arg_parser.parse_args()

    if args.images:
        image_stream = FileImageStream(args.images)
    else:
        image_stream = VideoStream(args.camera)

    display = InferenceDisplay(image_stream)
    display.run(args.model)


if __name__ == "__main__":
    main()
