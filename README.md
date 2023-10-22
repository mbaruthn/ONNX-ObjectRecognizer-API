# Model Training

To train the model to be used with appropriate labels on the CustomVision service,
you need to host it within the model folder and write the labels into the classes array.
The output model from CustomVision should be in the format of OnnxF16, GeneralCompact[S1].

# ONNX-ObjectDetectionAIApi

It performs inference on a single photo within the payload,
detecting the objects defined.

# Generating API Application Output

To obtain the output of a Windows application, you can use pyinstaller. After making all the necessary changes and conducting your API tests, you can prepare your application by running the following command:
```
pyinstaller --onefile --add-data "C:\Users\sadfgasdfs\openvino_env\Lib\site-packages\openvino\libs;." --noconsole --add-data "[{C:\Users\sadfgasdfs\AppData\Local\Programs\Python\}]Python39\Lib\site-packages\openvino;." --add-binary "C:\Users\sadfgasdfs\AppData\Local\Programs\Python\Python39\Lib\site-packages\openvino\inference_engine\constants.pyd;."E:\docker\src\vision\app.py

```
This command will create a ready-to-use .exe application.

## Performance Benchmark

- CPU: Intel Core i7-8700 
- 2 images in each request

| Method | Float Precision | Platform | Average Latency | Throughput |
|---|---|---|---|---|
| Batch Processing | FP16 | ONNX | 105 ms | 19 FPS |
| Sequential Processing (Default) | FP32 | ONNX | 71 ms | 28 FPS |
| Batch Processing | FP32 | ONNX | 67 ms | 30 FPS |

<--------------------------------------------------------------------------------------------------------------->

# Model Eğitimi

Kullanılacak olan modeli CustumVision servisi üzerinde uygun etiketler ile eğitimini yapıp, 
model klasörü içerisinde barındırmanız ve etiketleri classes arrayı içerisine yazmanız yeterli olacaktır.
CustumVision üzerinden OnnxF16, GeneralCompact[S1] model çıktısı almanız gerekmektedir.

# ONNX-ObjectDetectionAIApi

Payload içerisindeki tek bir fotoğraf üzerinde Inference yapar.
Tanımlı olan nesneleri bulur.

# API Uygulama Çıktısı Hazırlama

Bir windows uygulaması çıktısı almak için pyinstaller kullanabilirsiniz.
Tüm değişiklikleri ve API denemelerinizi gerçekleştirdikten sonra,
```
pyinstaller --onefile --add-data "C:\Users\sadfgasdfs\openvino_env\Lib\site-packages\openvino\libs;." --noconsole --add-data "[{C:\Users\sadfgasdfs\AppData\Local\Programs\Python\}]Python39\Lib\site-packages\openvino;." --add-binary "C:\Users\sadfgasdfs\AppData\Local\Programs\Python\Python39\Lib\site-packages\openvino\inference_engine\constants.pyd;."E:\docker\src\vision\app.py

```
komutu ile .exe uygulamanızı hazır hale getirebilirsiniz.


## Performace Benchmark

- CPU: Intel Core i7-8700 
- 2 images in each request

| Method | Float Precision | Platform | Average Latency | Throughput |
|---|---|---|---|---|
| Batch Processing | FP16 | ONNX | 105 ms | 19 FPS |
| Sequential Processing (Default) | FP32 | ONNX | 71 ms | 28 FPS |
| Batch Processing | FP32 | ONNX | 67 ms | 30 FPS |
