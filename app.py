# # import streamlit as st
# # import numpy as np
# # import tensorflow as tf
# # import json
# # from PIL import Image
# # import cv2

# # st.set_page_config(page_title="Waste Classification Dashboard", layout="wide")

# # # ---------- Load Models ----------
# # # @st.cache_resource
# # # def load_models():
# # #     mobilenet = tf.keras.models.load_model("mobilenetb0_model_v2.keras")
# # #     vgg = tf.keras.models.load_model("vgg16_model_v2.keras")
# # #     resnet = tf.keras.models.load_model("resnet50_model_v2.keras")
# # #     efficient = tf.keras.models.load_model("efficientnetb0_model_v2.keras")
# # #     return mobilenet, vgg, resnet, efficient
# # # @st.cache_resource
# # # def load_models():
# # #     mobilenet = tf.keras.models.load_model("mobilenetv2_model_v2.keras")
# # #     vgg = tf.keras.models.load_model("vgg16_amodel_v2.keras")
# # #     resnet = tf.keras.models.load_model("resnet50_model_v2.keras")
# # #     efficient = tf.keras.models.load_model("efficientnetb0_model_v2.keras")
# # #     return mobilenet, vgg, resnet, efficient
# # @st.cache_resource
# # def load_models():
# #     mobilenet = tf.keras.models.load_model("mobilenetv2_model_v2.keras", compile=False)
# #     vgg = tf.keras.models.load_model("vgg16_model_v2.keras", compile=False)
# #     resnet = tf.keras.models.load_model("resnet50_model_v2.keras", compile=False)
# #     efficient = tf.keras.models.load_model("efficientnetb0_model_v2.keras", compile=False)
# #     return mobilenet, vgg, resnet, efficient

# # mobilenet, vgg, resnet, efficient = load_models()

# # # ---------- Load Classes ----------
# # with open("class_names.json") as f:
# #     class_names = list(json.load(f).keys())

# # # ---------- Image Preprocess ----------
# # def preprocess(img):
# #     img = img.resize((224,224))
# #     img = np.array(img)/255.0
# #     img = np.expand_dims(img,axis=0)
# #     return img

# # # ---------- Prediction ----------
# # def predict_all_models(img):
# #     preds = {}

# #     preds["MobileNetV2"] = mobilenet.predict(img)
# #     preds["VGG16"] = vgg.predict(img)
# #     preds["ResNet50"] = resnet.predict(img)
# #     preds["EfficientNetB0"] = efficient.predict(img)

# #     return preds

# # # ---------- Sidebar ----------
# # page = st.sidebar.selectbox(
# #     "Navigation",
# #     ["Home",
# #      "Image Classification",
# #      "Object Detection",
# #      "Model Performance",
# #      "Live Webcam Detection",
# #      "About"]
# # )

# # # =========================================================
# # # HOME PAGE
# # # =========================================================
# # if page == "Home":

# #     st.title("♻ Waste Classification & Detection System")

# #     st.write("""
# #     This application classifies and detects waste objects using Deep Learning models.

# #     **Key Features**
# #     - CNN based waste classification
# #     - YOLO object detection
# #     - Multi-model comparison
# #     - Interactive dashboard
# #     """)

# #     st.image("https://cdn.pixabay.com/photo/2017/02/01/22/02/recycling-2035092_1280.png")

# # # =========================================================
# # # IMAGE CLASSIFICATION PAGE
# # # =========================================================
# # elif page == "Image Classification":

# #     st.title("Image Classification")

# #     uploaded = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])

# #     if uploaded:

# #         image = Image.open(uploaded)
# #         st.image(image,width=300)

# #         img = preprocess(image)

# #         preds = predict_all_models(img)

# #         st.subheader("Model Predictions")

# #         for model_name,pred in preds.items():

# #             top5 = np.argsort(pred[0])[-5:][::-1]

# #             st.write(f"### {model_name}")

# #             for i in top5:

# #                 st.write(
# #                     class_names[i],
# #                     round(pred[0][i]*100,2),
# #                     "%"
# #                 )

# # # =========================================================
# # # OBJECT DETECTION PAGE (Placeholder for YOLO)
# # # =========================================================
# # elif page == "Object Detection":

# #     st.title("YOLO Object Detection")

# #     st.write("Upload image for object detection")

# #     uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# #     if uploaded:

# #         image = Image.open(uploaded)
# #         st.image(image)

# #         st.info("YOLO detection code can be integrated here")

# # # =========================================================
# # # MODEL PERFORMANCE PAGE
# # # =========================================================
# # elif page == "Model Performance":

# #     st.title("Model Comparison")

# #     import pandas as pd

# #     data = {
# #         "Model":["MobileNetV2","VGG16","ResNet50","EfficientNetB0"],
# #         "Accuracy":[0.42,0.38,0.30,0.32],
# #         "Speed(ms)":[25,40,50,45]
# #     }

# #     df = pd.DataFrame(data)

# #     st.dataframe(df)

# #     st.bar_chart(df.set_index("Model")["Accuracy"])

# # # =========================================================
# # # WEBCAM DETECTION PAGE
# # # =========================================================
# # elif page == "Live Webcam Detection":

# #     st.title("Live Webcam Detection")

# #     run = st.checkbox("Start Camera")

# #     frame_window = st.image([])

# #     cap = cv2.VideoCapture(0)

# #     while run:

# #         ret, frame = cap.read()

# #         if not ret:
# #             break

# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         frame_window.image(frame)

# #     cap.release()

# # # =========================================================
# # # ABOUT PAGE
# # # =========================================================
# # elif page == "About":

# #     st.title("Project Information")

# #     st.write("""
# #     **Dataset**  
# #     Waste classification dataset with 26 classes.

# #     **Models Used**
# #     - MobileNetV2
# #     - VGG16
# #     - ResNet50
# #     - EfficientNetB0

# #     **Technologies**
# #     - Python
# #     - TensorFlow / Keras
# #     - YOLO
# #     - Streamlit

# #     **Developer**
# #     Data Science Project (GUVI Program)
# #     """)

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import json
# from PIL import Image
# import cv2
# import pandas as pd

# st.set_page_config(page_title="Waste Classification Dashboard", layout="wide")

# # --------------------------------------------------
# # LOAD MODEL
# # --------------------------------------------------
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("mobilenetv2_model.h5", compile=False)
#     return model

# model = load_model()

# # --------------------------------------------------
# # LOAD CLASS NAMES
# # --------------------------------------------------
# with open("class_names.json") as f:
#     class_names = list(json.load(f).keys())

# # --------------------------------------------------
# # IMAGE PREPROCESS
# # --------------------------------------------------
# # def preprocess(img):
# #     img = img.resize((224,224))
# #     img = np.array(img) / 255.0
# #     img = np.expand_dims(img, axis=0)
# #     return img



# # --------------------------------------------------
# # PREDICTION
# # --------------------------------------------------
# def predict(img):
#     pred = model.predict(img)
#     return pred

# # --------------------------------------------------
# # SIDEBAR
# # --------------------------------------------------
# page = st.sidebar.selectbox(
#     "Navigation",
#     [
#         "Home",
#         "Image Classification",
#         "Object Detection",
#         "Model Performance",
#         "Live Webcam Detection",
#         "About"
#     ]
# )

# # ==================================================
# # HOME PAGE
# # ==================================================
# if page == "Home":

#     st.title("♻ Waste Classification & Detection System")

#     st.write("""
#     This application classifies waste images using a deep learning model.

#     **Key Features**
#     - CNN based waste classification
#     - YOLO object detection (can be integrated)
#     - Interactive dashboard
#     """)

#     st.image(
#         "https://cdn.pixabay.com/photo/2017/02/01/22/02/recycling-2035092_1280.png"
#     )

# # ==================================================
# # IMAGE CLASSIFICATION
# # ==================================================
# elif page == "Image Classification":

#     st.title("Image Classification")

#     uploaded = st.file_uploader(
#         "Upload Waste Image",
#         type=["jpg","jpeg","png"]
#     )

#     if uploaded:

#         image = Image.open(uploaded)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.image(image, caption="Uploaded Image", width=300)

#         img = preprocess(image)

#         pred = predict(img)

#         with col2:

#             st.subheader("Top Predictions")

#             top5 = np.argsort(pred[0])[-5:][::-1]

#             for i in top5:

#                 st.write(
#                     f"{class_names[i]} : {round(pred[0][i]*100,2)} %"
#                 )

# # ==================================================
# # OBJECT DETECTION
# # ==================================================
# elif page == "Object Detection":

#     st.title("YOLO Object Detection")

#     st.info("YOLO detection can be integrated here.")

# # ==================================================
# # MODEL PERFORMANCE
# # ==================================================
# elif page == "Model Performance":

#     st.title("Model Comparison")

#     data = {
#         "Model":["MobileNetV2","VGG16","ResNet50","EfficientNetB0"],
#         "Accuracy":[0.42,0.38,0.30,0.32],
#         "Speed(ms)":[25,40,50,45]
#     }

#     df = pd.DataFrame(data)

#     st.dataframe(df)

#     st.bar_chart(df.set_index("Model")["Accuracy"])

# # ==================================================
# # WEBCAM
# # ==================================================
# elif page == "Live Webcam Detection":

#     st.title("Live Webcam")

#     run = st.checkbox("Start Camera")

#     frame_window = st.image([])

#     cap = cv2.VideoCapture(0)

#     while run:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         frame_window.image(frame)

#     cap.release()

# # ==================================================
# # ABOUT
# # ==================================================
# elif page == "About":

#     st.title("Project Information")

#     st.write("""
#     **Dataset**  
#     Waste classification dataset with 26 classes.

#     **Model Used**
#     MobileNetV2

#     **Technologies**
#     - Python
#     - TensorFlow / Keras
#     - Streamlit
#     - YOLO (optional)

#     **Developer**
#     Data Science Project
#     """)


# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import json
# from PIL import Image
# import cv2
# import pandas as pd

# st.set_page_config(page_title="Waste Classification Dashboard", layout="wide")

# # --------------------------------------------------
# # LOAD MODEL
# # --------------------------------------------------
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("mobilenetv2_model.h5", compile=False)
#     return model

# model = load_model()

# # --------------------------------------------------
# # LOAD CLASS NAMES
# # --------------------------------------------------
# with open("class_names.json") as f:
#     class_names = list(json.load(f).keys())

# # --------------------------------------------------
# # IMAGE PREPROCESS
# # --------------------------------------------------
# def preprocess(img):

#     img = img.convert("RGB")  # Fix RGBA issue

#     img = img.resize((224,224))

#     img = np.array(img).astype("float32") / 255.0

#     img = np.expand_dims(img, axis=0)

#     return img

# # --------------------------------------------------
# # PREDICTION
# # --------------------------------------------------
# def predict(img):

#     pred = model.predict(img)

#     return pred

# # --------------------------------------------------
# # SIDEBAR NAVIGATION
# # --------------------------------------------------
# page = st.sidebar.selectbox(
#     "Navigation",
#     [
#         "Home",
#         "Image Classification",
#         "Object Detection",
#         "Model Performance",
#         "Live Webcam Detection",
#         "About"
#     ]
# )

# # ==================================================
# # HOME PAGE
# # ==================================================
# if page == "Home":

#     st.title("♻ Waste Classification & Detection System")

#     st.write("""
#     This application classifies waste images using a deep learning model.

#     **Key Features**
#     - CNN based waste classification
#     - YOLO object detection (optional)
#     - Interactive dashboard
#     """)

#     st.image(
#         "https://cdn.pixabay.com/photo/2017/02/01/22/02/recycling-2035092_1280.png"
#     )

# # ==================================================
# # IMAGE CLASSIFICATION
# # ==================================================
# elif page == "Image Classification":

#     st.title("Waste Image Classification")

#     uploaded = st.file_uploader(
#         "Upload Waste Image",
#         type=["jpg","jpeg","png"]
#     )

#     if uploaded:

#         image = Image.open(uploaded)

#         # Image warnings
#         if image.mode != "RGB":
#             st.warning("Image converted to RGB format automatically.")

#         if image.size != (224,224):
#             st.info("Image will be resized to 224 × 224 for prediction.")

#         col1, col2 = st.columns(2)

#         with col1:

#             st.image(image, caption="Uploaded Image", width=300)

#             st.write("Image Size:", image.size)
#             st.write("Image Mode:", image.mode)

#         img = preprocess(image)

#         pred = predict(img)

#         with col2:

#             st.subheader("Top Predictions")

#             top5 = np.argsort(pred[0])[-5:][::-1]

#             chart_data = []

#             for i in top5:

#                 label = class_names[i]
#                 confidence = round(pred[0][i]*100,2)

#                 st.write(f"{label} : {confidence} %")

#                 chart_data.append({
#                     "Class": label,
#                     "Confidence": confidence
#                 })

#             df = pd.DataFrame(chart_data)

#             st.bar_chart(df.set_index("Class"))

# # ==================================================
# # OBJECT DETECTION
# # ==================================================
# elif page == "Object Detection":

#     st.title("YOLO Object Detection")

#     st.info("YOLO detection can be integrated here.")

#     uploaded = st.file_uploader(
#         "Upload Image for Detection",
#         type=["jpg","jpeg","png"]
#     )

#     if uploaded:

#         image = Image.open(uploaded)

#         st.image(image)

#         st.write("Detection output will appear here after YOLO integration.")

# # ==================================================
# # MODEL PERFORMANCE
# # ==================================================
# elif page == "Model Performance":

#     st.title("Model Comparison")

#     data = {
#         "Model":["MobileNetV2","VGG16","ResNet50","EfficientNetB0"],
#         "Accuracy":[0.42,0.38,0.30,0.32],
#         "Speed(ms)":[25,40,50,45]
#     }

#     df = pd.DataFrame(data)

#     st.dataframe(df)

#     st.bar_chart(df.set_index("Model")["Accuracy"])

# # ==================================================
# # LIVE WEBCAM
# # ==================================================
# elif page == "Live Webcam Detection":

#     st.title("Live Webcam Detection")

#     run = st.checkbox("Start Camera")

#     frame_window = st.image([])

#     cap = cv2.VideoCapture(0)

#     while run:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         frame_window.image(frame)

#     cap.release()

# # ==================================================
# # ABOUT PAGE
# # ==================================================
# elif page == "About":

#     st.title("Project Information")

#     st.write("""
#     **Dataset**  
#     Waste classification dataset with 26 classes.

#     **Model Used**
#     MobileNetV2

#     **Technologies**
#     - Python
#     - TensorFlow / Keras
#     - Streamlit
#     - YOLO (optional)

#     **Developer**
#     Data Science Project
#     """)

import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import cv2
import pandas as pd
from ultralytics import YOLO

st.set_page_config(page_title="Waste Classification Dashboard", layout="wide")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    # CNN classification model
    mobilenet = tf.keras.models.load_model(
        "mobilenetv2_model.h5",
        compile=False
    )

    # YOLO detection model
    yolo = YOLO("best.pt")

    return mobilenet, yolo


mobilenet, yolo_model = load_models()


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open("class_names.json") as f:
    class_names = list(json.load(f).keys())


# --------------------------------------------------
# IMAGE PREPROCESS
# --------------------------------------------------

def preprocess(img):

    img = img.convert("RGB")

    img = img.resize((224,224))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

def classify(img):

    pred = mobilenet.predict(img)

    return pred


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Image Classification",
        "Object Detection",
        "Model Performance",
        "Live Webcam Detection",
        "About"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("♻ Waste Classification & Detection System")

    st.write("""
    This application classifies and detects waste objects using Deep Learning models.

    **Features**
    - CNN waste classification
    - YOLO object detection
    - Interactive dashboard
    """)

    st.image(
        "https://cdn.pixabay.com/photo/2017/02/01/22/02/recycling-2035092_1280.png"
    )


# ==================================================
# IMAGE CLASSIFICATION
# ==================================================

elif page == "Image Classification":

    st.title("Waste Image Classification")

    uploaded = st.file_uploader(
        "Upload Waste Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        if image.mode != "RGB":
            st.warning("Image converted to RGB automatically")

        if image.size != (224,224):
            st.info("Image will be resized to 224x224")

        col1, col2 = st.columns(2)

        with col1:

            st.image(image, caption="Uploaded Image", width=300)

            st.write("Image Size:", image.size)

            st.write("Image Mode:", image.mode)

        img = preprocess(image)

        pred = classify(img)

        with col2:

            st.subheader("Top Predictions")

            top5 = np.argsort(pred[0])[-5:][::-1]

            chart_data = []

            for i in top5:

                label = class_names[i]

                confidence = round(pred[0][i]*100,2)

                st.write(f"{label} : {confidence} %")

                chart_data.append({
                    "Class":label,
                    "Confidence":confidence
                })

            df = pd.DataFrame(chart_data)

            st.bar_chart(df.set_index("Class"))


# ==================================================
# OBJECT DETECTION (YOLO)
# ==================================================
elif page == "Object Detection":

    st.title("YOLO Object Detection")

    uploaded = st.file_uploader(
        "Upload Image for Detection",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(image, caption="Uploaded Image")

        img = np.array(image)

        results = yolo_model(img)

        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result")


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("Model Comparison")

    data = {
        "Model":["MobileNetV2","VGG16","ResNet50","EfficientNetB0"],
        "Accuracy":[0.42,0.38,0.30,0.32],
        "Speed(ms)":[25,40,50,45]
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.bar_chart(df.set_index("Model")["Accuracy"])


# ==================================================
# WEBCAM
# ==================================================

elif page == "Live Webcam Detection":

    st.title("Live Webcam")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_window.image(frame)

    cap.release()


# ==================================================
# ABOUT
# ==================================================

elif page == "About":

    st.title("Project Information")

    st.write("""
    **Dataset**
    Waste classification dataset with 26 classes.

    **Models Used**
    - MobileNetV2 (classification)
    - YOLO (object detection)

    **Technologies**
    - Python
    - TensorFlow
    - YOLO
    - Streamlit
    """)