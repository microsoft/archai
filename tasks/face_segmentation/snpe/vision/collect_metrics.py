# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import cv2
import numpy as np
import os
import tqdm
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from PIL import Image

# Check the outputs of the Mask C-RNN model inference


def _get_dataset_image(filename, image_shape, dataset):
    inp_f = os.path.splitext(filename)[0] + ".png"
    img_file = os.path.join(dataset, inp_f)
    if not os.path.isfile(img_file):
        print(f"### dataset {img_file} not found")
        sys.exit(1)

    img = cv2.imread(img_file)[..., ::-1]     # BGR to RGB
    img = cv2.resize(img, image_shape, interpolation=cv2.INTER_LINEAR)
    img = img[..., ::-1]  # BGR to RGB
    return img


def _get_dataset_gt(img_name, dataset, img_shape, use_pillow=False):
    seg_name = img_name + '_seg.png'
    gt_f = os.path.join(dataset, seg_name)
    if not os.path.isfile(gt_f):
        print(f"### ground truth {gt_f} not found")
        sys.exit(1)

    gt_seg = cv2.imread(gt_f, cv2.IMREAD_GRAYSCALE)
    if gt_seg.shape[:2] != img_shape:
        if use_pillow:
            img = Image.fromarray(gt_seg, 'L')
            img = img.resize(img_shape[:2], Image.NEAREST)
            gt_seg = np.array(img)
        else:
            # cv2 resize is (newHeight, newWidth)
            newsize = [img_shape[1], img_shape[0]]
            gt_seg = cv2.resize(gt_seg, newsize, interpolation=cv2.INTER_NEAREST)
    return gt_seg


def show_output(input_shape, transpose, dataset, outputs):
    _, w, h, c = input_shape
    img_shape = (w, h)
    output_list = [x for x in os.listdir(outputs) if x.endswith('.raw')]
    output_list.sort()
    for out_f in output_list:
        img = _get_dataset_image(out_f, img_shape, dataset)
        logits = np.fromfile(os.path.join(outputs, out_f), dtype=np.float32)

        if (transpose):
            logits = logits.reshape((-1, img_shape[0], img_shape[1])).transpose(transpose)
        else:
            logits = logits.reshape((img_shape[0], img_shape[1], -1))

        cls_seg = np.argmax(logits, axis=-1)

        # debug visualize
        norm = cv2.normalize(cls_seg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cls_seg_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # concatenate on x-axis so result is 512 wide and 256 high.
        canvas = np.concatenate([img, cls_seg_color], axis=1)

        cv2.imshow('img', canvas)
        key = cv2.waitKey() & 0xFF
        if key == 27:
            break


def softmax(x, axis):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def normalize(x, axis):
    return x / np.expand_dims(np.linalg.norm(x, axis=axis), axis=axis)


def get_confusion_matrix(gt_label, pred_label, valid_mask, num_classes):
    assert gt_label.dtype in [np.int32, np.int64]
    assert pred_label.dtype in [np.int32, np.int64]
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index[valid_mask].flat, minlength=num_classes * num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes))

    yy, xx = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing='ij')
    ii = yy * num_classes + xx
    confusion_matrix[yy, xx] = label_count[ii]
    return confusion_matrix


def get_metrics(input_shape, transpose, dataset, outputs, num_classes=19, use_pillow=False):

    output_list = [x for x in os.listdir(outputs) if x.endswith('.raw')]
    output_list.sort()
    if len(output_list) == 0:
        print("No output files matching 'outputs/*.raw' found")
        return

    print(f"Collecting metrics on {len(output_list)} output .raw files...")

    width, height, c = input_shape
    img_shape = (width, height)
    confusion_matx = None

    bins = int(1e6)
    pos_by_score = np.zeros((num_classes, bins + 1))
    neg_by_score = np.zeros((num_classes, bins + 1))
    with tqdm.tqdm(total=len(output_list)) as pbar:
        for out_f in output_list:
            img_name = os.path.splitext(os.path.basename(out_f))[0].split('.')[0]
            gt_seg = _get_dataset_gt(img_name, dataset, img_shape, use_pillow)
            ignore_mask = (gt_seg == 255)
            gt_seg[ignore_mask] = 0
            gt_seg = gt_seg.astype(np.int32)
            valid_mask = np.logical_not(ignore_mask)

            full_path = os.path.join(outputs, out_f)
            logits = np.fromfile(full_path, dtype=np.float32)
            size = np.product(logits.shape)
            found_classes = int(size / (img_shape[0] * img_shape[1]))
            if found_classes != num_classes:
                raise Exception(f"Result {out_f} has unexpected number of predictions {found_classes}, " +
                                f"expecting {num_classes}")

            if transpose:
                logits = logits.reshape((num_classes, img_shape[0], img_shape[1])).transpose(transpose)
            else:
                logits = logits.reshape((img_shape[0], img_shape[1], num_classes))

            probs = softmax(logits.astype(np.float64), axis=-1)
            pd_seg = np.argmax(probs, axis=-1)

            # debug visualize
            # gt_seg_color = cv2.applyColorMap((255 * gt_seg / 19).astype(np.uint8), cv2.COLORMAP_JET)
            # pd_seg_color = cv2.applyColorMap((255 * pd_seg / 19).astype(np.uint8), cv2.COLORMAP_JET)
            # canvas = np.concatenate([gt_seg_color, pd_seg_color], axis=1)
            # cv2.imshow('img', canvas)
            # cv2.waitKey(0)
            matrix = get_confusion_matrix(gt_seg, pd_seg, valid_mask, num_classes)
            if confusion_matx is None:
                confusion_matx = matrix
            else:
                confusion_matx += matrix

            scores = (probs * bins).round().astype(np.int32)     # (b, h, w, num_classes)
            for c in range(num_classes):
                cls_mask = np.logical_and(gt_seg == c, valid_mask)     # (b, h, w)
                cls_score = scores[..., c]      # (b, h, w)

                pos_by_score[c] += np.bincount(cls_score[cls_mask], minlength=bins + 1)
                neg_by_score[c] += np.bincount(cls_score[np.logical_not(cls_mask)], minlength=bins + 1)

            pbar.update(1)

    class_names = ['background', 'skin', 'nose', 'right_eye', 'left_eye', 'right_brow', 'left_brow', 'right_ear',
                   'left_ear', 'mouth_interior', 'top_lip', 'bottom_lip', 'neck', 'hair', 'beard', 'clothing',
                   'glasses', 'headwear', 'facewear']
    assert len(class_names) == 19

    # compute iou and f1
    gt_pos = confusion_matx.sum(1)      # (num_classes,)
    pd_pos = confusion_matx.sum(0)      # (num_classes,)
    tp = np.diag(confusion_matx)        # (num_classes,)
    iou = tp / np.maximum(1, gt_pos + pd_pos - tp)      # (num_classes,)
    f1 = 2 * tp / np.maximum(1, gt_pos + pd_pos)      # (num_classes,)

    # compute weighted iou/f1 (excluding background and facewear class)
    weight = 1 / np.sqrt(gt_pos[1:18])
    if len(iou) > 1:
        overall_iou = np.sum(iou[1:18] * weight) / np.sum(weight)
        overall_f1 = np.sum(f1[1:18] * weight) / np.sum(weight)
    else:
        overall_iou = iou[0]
        overall_f1 = f1[0]

    # compute precision recall curve
    _, ax = plt.subplots(figsize=(6, 7))
    AP = []
    for c in range(num_classes):
        # get per class total count and total positives, sorted in score descending order
        cls_neg = neg_by_score[c][::-1]
        cls_pos = pos_by_score[c][::-1]

        tps = np.cumsum(cls_pos)
        fps = np.cumsum(cls_neg)

        precision = tps / np.maximum(1, tps + fps)
        # assert np.all(np.logical_not(np.isnan(precision)))
        # precision[np.isnan(precision)] = 0
        if tps[-1] == 0:
            recall = tps / 0.0000001
        else:
            recall = tps / tps[-1]

        # stop when full recall attained
        # and reverse the outputs so recall is decreasing
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        precision = np.r_[precision[sl], 1]
        recall = np.r_[recall[sl], 0]
        average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        AP.append(average_precision)

        # draw figure
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision,
            average_precision=average_precision,
        )
        display.plot(ax=ax, name=class_names[c])

    handles, labels = display.ax_.get_legend_handles_labels()
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Precision recall curve")
    chart = os.path.join(outputs, 'pr_curve.png')
    plt.savefig(chart)
    plt.close()  # fixes a huge memory leak

    # save metrics
    csv_file = os.path.join(outputs, 'test_results.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        df = pd.DataFrame(np.stack([iou, f1, AP], axis=0), columns=class_names[:num_classes], index=['iou', 'f1', 'AP'])
        df.loc[:, 'overall'] = pd.Series([overall_iou, overall_f1], index=['iou', 'f1'])
        df.to_csv(f)
        print(df)
    return (csv_file, chart, float(overall_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the outputs of the Mask C-RNN model inference and produce ' +
                                     'a .csv file named test_results.csv and a .png plot named pr_curve.png')
    parser.add_argument('--input', help='Location of the original input images ' +
                        '(defaults to INPUT_DATASET environment variable)')
    parser.add_argument('--show', '-s', help='Show the outputs on screen, press space to advance to the next image ' +
                        'and escape to cancel', action="store_true")
    parser.add_argument('--output', '-o', help='Location of the outputs to analyze (default "snpe_output")',
                        default='snpe_output')
    parser.add_argument('--transpose', '-t', help='Transpose channels by (1,2,0)', action="store_true")
    parser.add_argument('--num_classes', type=int, help="Number of classes predicted (default 19)", default=19)
    parser.add_argument('--pillow', help="Resize images using Pillow instead of numpy", action="store_true")
    parser.add_argument('--input_shape', help="Resize images this size, must match the shape of the model output " +
                                              "(default '256,256,3')")
    args = parser.parse_args()

    use_pillow = args.pillow
    dataset = args.input
    if not dataset:
        dataset = os.getenv("INPUT_DATASET")
        if not dataset:
            print("please provide --input or set your INPUT_DATASET environment vairable")
            sys.exit(1)

    transpose = args.transpose
    if transpose:
        transpose = (1, 2, 0)

    if not os.path.isdir(dataset):
        print("input dataset not found: " + dataset)
        sys.exit(1)

    output_dir = args.output
    if not os.path.isdir(output_dir):
        print("Experiment 'output' dir not found: " + output_dir)
        sys.exit(1)

    input_shape = (256, 256, 3)
    if args.input_shape:
        input_shape = tuple(eval(args.image_shape))

    if args.show:
        show_output(input_shape, transpose, dataset, output_dir)
    else:
        get_metrics(input_shape, transpose, dataset, output_dir, args.num_classes, use_pillow)
