#!/usr/bin/env python3
"""
Uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


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
            input_width = int(self.model.input.shape[1])
            input_height = int(self.model.input.shape[2])
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_scores_full = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores_full.append(box_conf * box_class_prob)

        # box_scores
        box_scores_list = [score.max(axis=3) for score in box_scores_full]
        box_scores_list = [score.reshape(-1) for score in box_scores_list]
        box_scores = np.concatenate(box_scores_list)
        index_to_delete = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, index_to_delete)

        # box_classes
        box_classes_list = [box.argmax(axis=3) for box in box_scores_full]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        # filtered_boxes
        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou_calc(box1, box2):
        """
        Intersection over union
        (x1, y1, x2, y2)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

        if w_intersection <= 0 or h_intersection <= 0:
            return 0

        intersection = w_intersection * h_intersection
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        if len(filtered_boxes) == 0:
            return []

        i_sorted = np.lexsort((-box_scores, box_classes))
        sorted_len = len(i_sorted)

        for idx in range(sorted_len - 1):
            i = idx + 1
            suppress = []
            if i < len(i_sorted):
                while (box_classes[i_sorted[idx]] ==
                       box_classes[i_sorted[i]]):
                    iou = self.iou_calc(filtered_boxes[i_sorted[idx]],
                                        filtered_boxes[i_sorted[i]])
                    if iou > self.nms_t:
                        suppress.append(i)
                    i += 1
                    if i >= len(i_sorted):
                        break
                idx = i
            i_sorted = np.delete(i_sorted, suppress)
        return (filtered_boxes[i_sorted],
                box_classes[i_sorted],
                box_scores[i_sorted])

    @staticmethod
    def load_images(folder_path):
        """
        Returns a tuple of (images, image_paths)
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(image) for image in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Returns a tuple of (pimages, image_shapes)
        """
        input_w = int(self.model.input.shape[1])
        input_h = int(self.model.input.shape[2])

        pimages_list = []
        image_shapes_list = []

        for img in images:
            # save original image size
            img_shape = img.shape[0], img.shape[1]
            image_shapes_list.append(img_shape)

            # Resize the images with inter-cubic interpolation
            dim = (input_w, input_h)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

            # Rescale all images to have pixel values in the range [0, 1]
            pimage = resized / 255
            pimages_list.append(pimage)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names,
        and box scores
        """
        for i in range(len(boxes)):
            # Box scores should be rounded to 2 decimal places
            score = " {:.2f}".format(box_scores[i])

            # BOXES
            # Top left corner of image
            start_point = (int(boxes[i, 0]), int(boxes[i, 1]))
            # Bottom right corner of image
            end_point = (int(boxes[i, 2]), int(boxes[i, 3]))
            # Blue color in BGR
            color = (255, 0, 0)

            thickness = 2

            image = cv2.rectangle(image,
                                  start_point, end_point,
                                  color, thickness)

            # Text should be written 5 pixels (it was changed to 10 pixels
            # for better readability) above the top left corner of the box.
            # org is the Bottom-left corner of the text string in the image.
            org = (int(boxes[i, 0]), int(boxes[i, 1] - 10))
            # fontScale was changed from 0.5 to 0.9 for better readability
            fontScale = 0.9

            # Red color in BGR
            color = (0, 0, 255)

            # thickness was changed from 1 to 2 for better readability
            thickness = 2
            image = cv2.putText(image,
                                self.class_names[box_classes[i]] + score,
                                org, cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, color, thickness,
                                cv2.LINE_AA)

        # Show image
        cv2.imshow(file_name, image)

        # Display the window infinitely until any keypress
        key = cv2.waitKey(0)
        if key == ord('s'):
            # If 'detections' directory  does not exist, create it
            os.mkdir('detections') if not os.path.isdir('detections') else None

            # Change the current directory
            os.chdir('detections')

            # Save the image
            cv2.imwrite(file_name, image)

            # Change back to working directory
            os.chdir('../')
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Returns: a tuple of (predictions, image_paths)
        """
        predictions = []

        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        for i in range(pimages.shape[0]):
            current_out = [out[i] for out in outputs]

            boxes, box_confidences, box_class_probs = \
                self.process_outputs(current_out, image_shapes[i])

            filtered_boxes, box_classes, box_scores = \
                self.filter_boxes(boxes, box_confidences, box_class_probs)

            box_predictions, predicted_box_classes, predicted_box_scores = \
                self.non_max_suppression(filtered_boxes,
                                         box_classes,
                                         box_scores)

            # All image windows should be named
            # after the corresponding image filename without its full path
            file_name = image_paths[i].split('/')[-1]
            # Displays all images using the show_boxes method
            self.show_boxes(images[i], box_predictions,
                            predicted_box_classes,
                            predicted_box_scores,
                            file_name)

            predictions.append((box_predictions,
                                predicted_box_classes,
                                predicted_box_scores))

        return predictions, image_paths
