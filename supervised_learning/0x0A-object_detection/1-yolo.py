#!/usr/bin/env python3
"""
Uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Sigmoid function
        """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Returns a tuple of (boxes, box_confidences, box_class_probs)
        """
        # Image’s original size
        image_height, image_width = image_size[0], image_size[1]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            # list with processed boundary boxes for each output
            boxes.append(output[..., 0:4])
            # list with box confidences for each output
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            # list with box's class probabilities for each output
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        for count, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)
            # cx, cy: cell's top left corner of the box
            cy = c + idx_y
            cx = c + idx_x

            # The network predicts 4 coordinates for each bounding box
            t_x = (box[..., 0])
            t_y = (box[..., 1])

            # normalize the above variables
            t_x_n = self.sigmoid(t_x)
            t_y_n = self.sigmoid(t_y)

            # width and height
            t_w = (box[..., 2])
            t_h = (box[..., 3])

            """
            If the cell is offset from the top left corner of the
            image by (cx, cy) and the bounding box prior has width and
            height pw, ph, then the predictions correspond to
            """

            # center
            bx = t_x_n + cx
            by = t_y_n + cy

            # normalization
            bx /= grid_width
            by /= grid_height

            # priors (anchors) width and height
            pw = self.anchors[count, :, 0]
            ph = self.anchors[count, :, 1]

            # scale to anchors box dimensions
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # normalize to model input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # Corners of bounding box
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # scale to image original size
            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height

        return boxes, box_confidences, box_class_probs
