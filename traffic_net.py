# // Copyright (c) 2025 VulcanYJX
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at

# //     http://www.apache.org/licenses/LICENSE-2.0

# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import onnxruntime as ort
import numpy as np
import cv2


class TrafficNet:
    """
    Implements inference for an ONNX model.
    """

    def __init__(self, model_path):
        """
        :param model_path: The path to the ONNX model to load from disk.
        """
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        # print( self.input_shape )
        self.classes = ["car", "bicycle", "person", "road_sign"]

    def preprocess_img(self, img, input_size=[1, 3, 544, 960]):
        """
        Preprocess the input image to match the ONNX model requirements.
        :param img: Original image as a numpy array.
        :param input_size: Target input size of the model (e.g., [1, 3, H, W]).
        :return: Preprocessed image as a numpy array.
        """
        # Resize and normalize the image
        img_resized = cv2.resize(img, (input_size[3], input_size[2]))
        img_normalized = img_resized / 255.0  # Normalize to [0, 1]
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batched = np.expand_dims(img_transposed, axis=0).astype(
            np.float32
        )  # Add batch dimension
        return img_batched

    def preprocess_image_with_padding(self, image, target_width=960, target_height=544):
        """
        Preprocesses an image for input into the NVIDIA DetectNet_v2 model by maintaining
        the aspect ratio and padding to fit the target size.

        Parameters:
            image (numpy.ndarray): The input image in RGB format.
            target_width (int): Target width of the image (default: 960).
            target_height (int): Target height of the image (default: 544).

        Returns:
            numpy.ndarray: Preprocessed image in NCHW format (Batch size = 1).
        """
        # Step 1: Get original dimensions of the image
        original_height, original_width, _ = image.shape

        # Step 2: Calculate scale factor to fit the target size while maintaining aspect ratio
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)

        # Step 3: Resize the image with the calculated dimensions
        resized_image = cv2.resize(image, (resized_width, resized_height))

        # Step 4: Create a blank canvas with target dimensions and fill with zeros (black padding)
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.float32)

        # Step 5: Place the resized image in the center of the canvas
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        padded_image[top : top + resized_height, left : left + resized_width, :] = (
            resized_image / 255.0
        )  # Normalize here

        # Step 6: Convert the image to NCHW format
        chw_image = np.transpose(padded_image, (2, 0, 1))
        nchw_image = np.expand_dims(chw_image, axis=0)  # Add batch dimension (N=1)

        return nchw_image

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A preprocessed batch of images as a numpy array.
        :return: Raw outputs from the ONNX model.
        """
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        return outputs

    @staticmethod
    def divide_and_round_up(a, b):
        return (a + b - 1) // b

    @staticmethod
    def clip(value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def nms_numpy(self, boxes, scores, nms_threshold):
        """
        Perform Non-Maximum Suppression (NMS) using NumPy.

        Args:
            boxes (np.ndarray): Array of bounding boxes with shape (N, 4), where each box is [x1, y1, x2, y2].
            scores (np.ndarray): Array of scores/confidences with shape (N,).
            nms_threshold (float): Threshold for IoU to filter overlapping boxes.

        Returns:
            list: Indices of boxes to keep after NMS.
        """
        if len(boxes) == 0:
            return []

        # Compute areas of all bounding boxes
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by confidence score in descending order
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]  # Index of the box with the highest score
            keep.append(i)

            # Compute IoU of the highest-score box with the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Compute width and height of overlaps
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute IoU
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union

            # Keep boxes with IoU below the threshold
            remaining = np.where(iou <= nms_threshold)[0]
            order = order[remaining + 1]

        return keep

    def getBbox(self, frame, detections, nms_threshold=0.1):
        """
        Draw bounding boxes on the frame with detections after applying Non-Maximum Suppression (NMS).

        Args:
            frame (np.ndarray): The input frame (image) on which to draw.
            detections (list): A list of dictionaries with keys 'class', 'bbox', and 'confidence'.
            nms_threshold (float): Threshold for NMS to filter overlapping boxes.

        Returns:
            np.ndarray: The frame with bounding boxes drawn.
        """
        # Extract detection details
        labels = np.array([det["class"] for det in detections])
        bounding_boxes = np.array([det["bbox"] for det in detections])
        confs = np.array([det["confidence"] for det in detections])

        # Perform NMS using NumPy
        keep_idx = self.nms_numpy(bounding_boxes, confs, nms_threshold)

        # Filter detections based on NMS results
        bounding_boxes = bounding_boxes[keep_idx]
        confs = confs[keep_idx]
        labels = labels[keep_idx]

        if len(keep_idx) == 1:
            bounding_boxes = np.expand_dims(bounding_boxes, axis=0)
            confs = np.expand_dims(confs, axis=0)
            labels = np.expand_dims(labels, axis=0)

        # Bounding box scale factors
        input_w, input_h = self.input_shape[3], self.input_shape[2]  # (960, 544)
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        w_sc = frame_w / input_w  # Scale width from model to frame
        h_sc = frame_h / input_h  # Scale height from model to frame
        scales = np.array([w_sc, h_sc, w_sc, h_sc])

        # Scale bounding boxes
        bounding_boxes = np.floor(bounding_boxes * scales).astype(int)

        # Draw bounding boxes
        # for i, bbox in enumerate(bounding_boxes):
        #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     cv2.putText(frame, self.classes[labels[i]], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # return frame
        return [bounding_boxes, labels]

    def postprocess(self, outputs, confidence=0.2):

        object_list = []
        cov_layer_index = 0
        cov_layer_dims = outputs[cov_layer_index].shape
        bbox_layer_index = 1
        bbox_layer_dims = outputs[bbox_layer_index].shape

        num_classes_to_parse = len(self.classes)
        grid_w, grid_h = cov_layer_dims[3], cov_layer_dims[2]
        bbox_norm_x, bbox_norm_y = 35.0, 35.0
        stride_x = self.divide_and_round_up(self.input_shape[3], bbox_layer_dims[3])
        stride_y = self.divide_and_round_up(self.input_shape[2], bbox_layer_dims[2])

        gc_centers_x = (np.arange(grid_w) * stride_x + 0.5) / bbox_norm_x
        gc_centers_y = (np.arange(grid_h) * stride_y + 0.5) / bbox_norm_y

        for cl in range(num_classes_to_parse):

            for y in range(grid_h):
                for x in range(grid_w):
                    cov = outputs[cov_layer_index][0][cl][y][x]
                    if cov < confidence:
                        continue
                    bbox = outputs[bbox_layer_index][0][cl * 4 : (cl + 1) * 4, y, x]

                    x_min = (bbox[0] - gc_centers_x[x]) * -bbox_norm_x
                    y_min = (bbox[1] - gc_centers_y[y]) * -bbox_norm_y
                    x_max = (bbox[2] + gc_centers_x[x]) * bbox_norm_x
                    y_max = (bbox[3] + gc_centers_y[y]) * bbox_norm_y

                    object_list.append(
                        {
                            "class": cl,
                            "confidence": cov,
                            "bbox": [
                                self.clip(x_min, 0, self.input_shape[3] - 1),
                                self.clip(y_min, 0, self.input_shape[2] - 1),
                                self.clip(x_max, 0, self.input_shape[3] - 1) + 1,
                                self.clip(y_max, 0, self.input_shape[2] - 1) + 1,
                            ],
                        }
                    )

        return object_list


def main():
    # Load ONNX model
    model_path = "resnet18_trafficcamnet_pruned.onnx"
    onnx_infer = TrafficNet(model_path)
    # Load and preprocess the image
    img = cv2.imread("000020_input.jpg")
    classes = ["car", "bicycle", "person", "road_sign"]

    preprocessed_img = onnx_infer.preprocess_image_with_padding(img)
    # Perform inference
    outputs = onnx_infer.infer(preprocessed_img)
    detect = onnx_infer.postprocess(outputs)
    [bboxes, labels] = onnx_infer.getBbox(img, detect)

    for i, bbox in enumerate(bboxes):
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            img,
            classes[labels[i]],
            (bbox[0], bbox[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    cv2.imwrite("res.jpg", img)
    print("ok")


# if __name__ == "__main__":
#     main()
