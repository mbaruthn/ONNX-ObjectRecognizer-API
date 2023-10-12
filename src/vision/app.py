import numpy as np
import onnx
import onnxruntime
import PIL.Image
import time
from aioflask import Flask, jsonify, request

# Flask uygulamasını oluşturma
app = Flask(__name__)

# Sabitler ve sınıf etiketleri
PROB_THRESHOLD = 0.1
classes = ["bottle", "bottle_hand", "other"]

# Model sınıfı oluşturma
class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image):
        image = image.resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

# Flask uygulaması başlatılırken modeli yükleyin
model = None
def runOD(model_filepath):
    global model
    model = Model(model_filepath)
    if model is None:
        print("Model NOT FOUND!")

# REST API endpoints
@app.route("/", methods=["GET"])
async def index():
    return jsonify(
        {
            "ServiceName": "Object Detection Vision API",
            "version": "0.1.0",
            "Environment": "Development",
        }
    )

@app.route("/", methods=["POST"])
async def main():
    global model
    image_files = request.files.getlist("image_files")
    response = {
        "results": None,
        "message": None,
    }

    if model is None:
        response["message"] = "Model not found"
    else:
        global elapsed
        filenames = []
        for image_file in image_files:
            filenames.append(image_file.filename)
            imageF = PIL.Image.open(image_file)
        
        start_time = time.time()  
        predictions = model.predict(imageF)
        end_time = time.time()
        
        elapsed = end_time - start_time
        results_list = []
        for i in range(len(filenames)):
            prediction_dict = {"filename": filenames[i], "predictions": []}
            assert set(predictions.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
            
            for class_id, score in zip(predictions['detected_classes'][0], predictions['detected_scores'][0]):
                if score > PROB_THRESHOLD:
                    classes_final = classes[class_id]
                    score_final = score
                    prediction_dict["predictions"].append({"tagName": classes_final, "probability": score_final})
            
            results_list.append(prediction_dict)
        response["results"] = results_list
        response["message"] = "OK"
        
    print(elapsed)
    return jsonify(response)

# Ana uygulamayı başlatma
if __name__ == "__main__":
    # Modelin bulunacağı sabit yolu belirtiyoruz
    model_filepath = "./model/model.onnx"
    runOD(model_filepath)
    app.run(debug=False, host="0.0.0.0", port=8100)
