from flask import Flask, request, jsonify
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from io import BytesIO
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
# import io
# from PIL import Image, ImageDraw, ImageFont
# import torchvision.transforms as T
# import torch
# from collections import Counter
# import base64

app = Flask(__name__)
CORS(app)

# =============================== KLASIFIKASI PENYAKIT PADI ===============================

cnn_model = load_model('cnn_model.h5')

img_width, img_height = 256, 256

labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Spot"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Tidak ada file yang dikirim"})

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({"error": "Nama file tidak valid"})

        try:
            img = Image.open(BytesIO(image_file.read()))
        except UnidentifiedImageError:
            return jsonify({"error": "File yang diunggah bukan gambar yang valid"})

        img = img.resize((img_width, img_height))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array /= 255.0

        probs = cnn_model.predict(img_array)
        probs = np.clip(probs, 0, 1)
        percent = probs * 100

        high = np.argmax(probs)
        label = labels[high]
        accuracy = np.round(percent[0, high], 2)

        return jsonify({
            "label": label,
            "accuracy": float(accuracy)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
# =============================== IDENTIFIKASI BERAS ===============================
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# faster_model = torch.load("faster_rcnn_model.pkl", map_location=device)
# faster_model.to(device)

# def identify_image(model, image, device, threshold=0.5):
#     model.eval()
    
#     transform = T.ToTensor()
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         predictions = model(image_tensor)
    
#     predictions = [{k: v.to("cpu") for k, v in pred.items()} for pred in predictions]
    
#     boxes = predictions[0]["boxes"][predictions[0]["scores"] > threshold].numpy()
    # scores = predictions[0]["scores"][predictions[0]["scores"] > threshold].numpy()
    # labels = predictions[0]["labels"][predictions[0]["scores"] > threshold].numpy()
    # # 
    # class_counts = Counter(labels)
    
#     class_0_count = class_counts.get(0, 0)
#     class_1_count = class_counts.get(1, 0)
#     class_2_count = class_counts.get(2, 0)
#     class_3_count = class_counts.get(3, 0)
#     total_objects = len(labels)
    
#     class_0_percentage = (class_0_count / total_objects) * 100 if total_objects > 0 else 0
#     class_1_percentage = (class_1_count / total_objects) * 100 if total_objects > 0 else 0
#     class_2_percentage = (class_2_count / total_objects) * 100 if total_objects > 0 else 0
#     class_3_percentage = (class_3_count / total_objects) * 100 if total_objects > 0 else 0
    
#     print(f"Persentase Benda asing: {class_0_percentage:.2f}%")
#     print(f"Persentase Butir kepala: {class_1_percentage:.2f}%")
#     print(f"Persentase Butir menir: {class_2_percentage:.2f}%")
#     print(f"Persentase Butir patah: {class_3_percentage:.2f}%")
    
#     if class_1_percentage >= 85:
#         if class_3_percentage >=14:
#             if class_2_percentage >= 1:
#                 if class_0_percentage >= 0.1:
#                     print("Beras Premium")
#     if class_1_percentage >= 80:
#         if class_3_percentage >=18:
#             if class_2_percentage >= 2:
#                 if class_0_percentage >= 0.2:
#                     print("Beras Medium 1")
#     if class_1_percentage >= 75:
#         if class_3_percentage >=22:
#             if class_2_percentage >= 3:
#                 if class_0_percentage >= 0.3:
#                     print("Beras Medium 2")
    
#     draw = ImageDraw.Draw(image)
#     font_path = "arial.ttf"  
#     font_size = 48 
#     font = ImageFont.truetype(font_path, font_size)
    
#     for box, score in zip(boxes, scores):
#         draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="cyan", width=8)
#         text_position = (box[0], box[1] - 50 if box[1] - 50 > 0 else box[1] + 50)
#         draw.text(text_position, f"{score:.3f}", fill="cyan", font=font)
    
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
#     class_counts = {int(k): v for k, v in class_counts.items()}
    
#     result = {
#         "boxes": boxes.tolist(),
#         "scores": scores.tolist(),
#         "labels": labels.tolist(),
#         "class_counts": class_counts,
#         "class_0_percentage": class_0_percentage,
#         "class_1_percentage": class_1_percentage,
#         "class_2_percentage": class_2_percentage,
#         "class_3_percentage": class_3_percentage,
#         "image_with_boxes": img_str,
#     }
    
#     return result

# @app.route('/identify', methods=['POST'])
# def identify():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     try:
#         image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         result = identify_image(faster_model, image, device)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
