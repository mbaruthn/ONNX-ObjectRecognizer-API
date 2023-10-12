import os
import tempfile
import numpy as np
import onnx, onnxruntime
from lib.object_detection import ObjectDetection

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(model_filename))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_images):
        model_inputs = []
        for preprocessed_image in preprocessed_images:
            inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
            inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
            model_inputs.append(inputs)
        
        model_inputs = np.concatenate(model_inputs, axis=0)
        if self.is_fp16:
            model_inputs = model_inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: model_inputs})[0]

        transformed_outs = []
        for i in range(outputs.shape[0]):
            output = outputs[i]
            output = np.squeeze(output).transpose((1,2,0)).astype(np.float32)
            transformed_outs.append(output)
        return transformed_outs