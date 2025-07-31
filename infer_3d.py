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

from traffic_net import TrafficNet
from pose3D_estimation import Pose3DEstimation
from utils.Math import *
from utils.Plotting import *

from utils import ClassAverages
import cv2
import numpy as np
from vedo import Plotter, load


class Infer3DBox:
    def __init__(self, traffic_onnx_path, pose_onnx_path, proj_matrix):
        self.traffic_net_infer = TrafficNet(traffic_onnx_path)
        self.pose_net_infer = Pose3DEstimation(pose_onnx_path)
        self.classes = ["car", "cyclist", "pedestrian", "road_sign"]
        self.input_shape = self.traffic_net_infer.input_shape

        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = self.generate_bins(2)
        if isinstance(proj_matrix, str):
            proj_matrix = get_P(proj_matrix)

        self.proj_matrix = proj_matrix

        car_model_path = "./config/car.obj"
        self.plotter = Plotter()
        self.car_model = load(car_model_path)

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """
        Calculate global angle of object, see paper
        """
        width = img.shape[1]
        # Angle of View: fovx (rad) => 3.14
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def plot3d(self, img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

        # the math! returns X, the corners used for constraint
        location, X = calc_location(
            dimensions, self.proj_matrix, box_2d, alpha, theta_ray
        )

        orient = alpha + theta_ray

        # magic number: 3.0
        locationXZY = np.array([location[0] * 3.0, location[2], location[1]])
        car = self.car_model.clone()

        # magic number: 0.5
        dimensions[2] = dimensions[2] * 0.5
        car.scale(dimensions)
        car.rotate(orient * 180 / np.pi, axis=(0, 0, 1))
        car.pos(locationXZY)
        car_color = "gray"
        car.color(car_color)
        self.plotter += car

        if img_2d is not None:
            plot_2d_box(img_2d, box_2d)

        plot_3d_box(img, self.proj_matrix, orient, dimensions, location)  # 3d boxes

        return location

    def show(self):
        self.plotter.show(title="3D Car Visualization", axes=1)

    def generate_bins(self, bins):
        angle_bins = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            angle_bins[i] = i * interval
        angle_bins += interval / 2  # center of bins
        return angle_bins

    def infer(self, frame):
        pre_traffic_img = self.traffic_net_infer.preprocess_image_with_padding(frame)
        detect_outs = self.traffic_net_infer.infer(pre_traffic_img)
        detect = self.traffic_net_infer.postprocess(detect_outs, 0.2)
        return detect

    def cutImage(self, frame, detect):
        [bboxes, labels] = self.traffic_net_infer.getBbox(frame, detect)
        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x_min = max(0, bbox[0])
                y_min = max(0, bbox[1])
                x_max = min(frame.shape[1], bbox[2])
                y_max = min(frame.shape[0], bbox[3])

                # 计算宽度和高度
                # width = x_max - x_min
                # height = y_max - y_min

                crop = frame[y_min:y_max, x_min:x_max]

                if crop is None or crop.size == 0:
                    continue
                try:
                    pose_img = self.pose_net_infer.preprocess_image(crop)
                except Exception as e:
                    print(f"预处理图像失败: {e}")
                    continue
                [orient, conf, dim] = self.pose_net_infer.infer_onnx_model(pose_img)
                orient = orient[0]
                conf = conf[0]
                dim = dim[0]
                box_2d = [(x_min, y_min), (x_max, y_max)]
                dim += self.averages.get_item(self.classes[labels[i]])

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += self.angle_bins[argmax]
                alpha -= np.pi
                theta_ray = self.calc_theta_ray(frame, box_2d, self.proj_matrix)
                self.plot3d(frame, box_2d, dim, alpha, theta_ray)
        return frame


def main(args=None):
    traffic_onnx_path = "resnet18_trafficcamnet_pruned.onnx"
    pose_onnx_path = "pose_3d.onnx"
    img = cv2.imread("./image/000175.png")
    infer_onnx = Infer3DBox(
        traffic_onnx_path, pose_onnx_path, "config/calib_cam_to_cam.txt"
    )
    detect_res = infer_onnx.infer(img)
    res_img = infer_onnx.cutImage(img, detect_res)
    cv2.imwrite("res_img.jpg", res_img)


if __name__ == "__main__":
    main()
