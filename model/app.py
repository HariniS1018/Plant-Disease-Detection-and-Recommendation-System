from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pickle
import cv2

app = FastAPI()

# Load your TensorFlow model (do this once, globally)
try:
    tflite_path = "D:/Projects/plant-disease-detection-recommendation-system/model/quantised_groundnut_disease_model.tflite"
    le_path = "/content/drive/MyDrive//Projects/Plant Disease Prediction/saved_models/93acc_label_mapping.pkl"

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(le_path, "rb") as f:
        loaded_mapping = pickle.load(f)
except (
    FileNotFoundError,
    pickle.UnpicklingError,
    ValueError,
) as e:
    print(f"Error loading model or mapping: {e}")
    interpreter = None
    loaded_mapping = None


def preprocess_image(image_content):
    try:
        nparr = np.frombuffer(image_content, np.uint8)  # Convert bytes to NumPy array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode using cv2.imdecode

        img = cv2.resize(img, (224, 224))

        original_shape = img.shape
        image_2d = img.reshape(-1, 3)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_2d = scaler.fit_transform(image_2d)
        normalized_image = normalized_2d.reshape(original_shape)
        normalized_image = np.expand_dims(normalized_image, axis=0)
        return normalized_image

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


def get_remedies(disease_name) -> list:
    remedies = []
    return remedies


@app.post("/predict_disease")
async def predict_disease(image: UploadFile = File(...)):
    if interpreter is None or loaded_mapping is None:
        return JSONResponse({"error": "Model or mapping not loaded"}, status_code=500)

    try:
        if image.content_type.startswith("image/"):  # Check if it's an image
            image_content = await image.read()
            processed_image = preprocess_image(image_content)

            if processed_image is None:
                return JSONResponse(
                    {"error": "Image preprocessing failed"}, status_code=400
                )

            interpreter.set_tensor(input_details[0]["index"], processed_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])
            probabilities = output_data[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)

            predicted_label = loaded_mapping[predicted_class]
            prediction = f"{predicted_label}, Confidence: {confidence * 100:.2f}%"
            print(prediction)

            remedies: list = get_remedies(disease_name=predicted_label)
            return JSONResponse(
                {
                    "predicted_label": predicted_label,
                    "confidence": float(confidence),
                    "remedies": remedies,
                }
            )
        else:
            return JSONResponse(
                {"error": "Invalid file type. Please upload an image."}, status_code=400
            )

    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="api_service_9:app", workers=1, host="0.0.0.0", port=8000)
