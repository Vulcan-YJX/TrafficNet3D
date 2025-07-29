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


class Pose3DEstimation:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image, input_size=(224, 224)):
        """
        预处理图片，将其调整为模型需要的输入格式，并添加均值和标准差归一化。

        :param image_path: 图片路径
        :param input_size: 模型要求的输入尺寸 (宽, 高)
        :return: 预处理后的图片数据，形状为 [1, 3, H, W]
        """
        # 均值和标准差 (ImageNet 数据集的标准值)
        mean = [0.485, 0.456, 0.406]  # 每个通道的均值 (B, G, R)
        std = [0.229, 0.224, 0.225]  # 每个通道的标准差 (B, G, R)

        # 2. 转换为 RGB 格式 (OpenCV 默认读取为 BGR 格式)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 调整图片大小
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)

        # 4. 转换为 numpy 数组并归一化到 [0, 1]
        image_data = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

        # 5. 标准化处理 (每个通道减去均值再除以标准差)
        # 注意：image_data 的形状是 [H, W, C]，需要逐通道进行处理
        image_data -= mean
        image_data /= std

        # 6. 转换为 NCHW 格式 [C, H, W]
        image_data = np.transpose(image_data, (2, 0, 1))

        # 7. 增加 batch 维度，形状变为 [1, 3, H, W]
        image_data = np.expand_dims(image_data, axis=0)

        return image_data

    def infer_onnx_model(self, image_data):
        """
        使用 ONNX 模型进行推理。
        :param onnx_model_path: ONNX 模型路径
        :param image_data: 预处理后的图片数据
        :return: 推理结果
        """
        # 加载 ONNX 模型

        # 获取输入和输出层名

        # 执行推理
        input_data = image_data.astype(np.float32)
        # result = session.run([output_name], {input_name: input_data})
        return self.session.run(None, {self.input_name: input_data})
