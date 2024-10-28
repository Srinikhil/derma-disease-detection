# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import numpy as np
# from PIL import Image
# import cv2

# # Load the fine-tuned VGG16 model
# @st.cache(allow_output_mutation=True)
# def load_vgg16_model():
#     model = load_model('derma_disease_detection_VGG16_fine_tuned_colab_20241026_013722.h5')
#     return model

# # Dictionary to map predictions to diseases and prescriptions
# disease_prescription = {
#     0: ("Eczema", "Apply a moisturizing cream like CeraVe twice a day."),
#     1: ("Psoriasis", "Apply a topical corticosteroid and keep skin moisturized."),
#     2: ("Acne", "Use a salicylic acid cleanser and apply benzoyl peroxide cream.")
# }

# # Set up the Streamlit app
# st.title("Skin Disease Detection and Prescription")

# # Upload an image
# uploaded_file = st.file_uploader("Upload an image of the skin condition", type=["jpg", "jpeg", "png"])

# # Display the uploaded image
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image to match the model's input size
#     img_array = img_to_array(image)
#     img_array = cv2.resize(img_array, (224, 224))  # VGG16 input size
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array /= 255.0  # Normalize the image

#     # Load the model
#     model = load_vgg16_model()

#     # Make a prediction
#     if st.button("Predict"):
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions)

#         # Display the result
#         disease, prescription = disease_prescription[predicted_class]
#         st.success(f"Prediction: {disease}")
#         st.info(f"Prescription: {prescription}")



###

# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import numpy as np
# from PIL import Image
# import cv2

# # Load the fine-tuned VGG16 model
# # @st.cache(allow_output_mutation=True)
# def load_vgg16_model():
#     st.write("Loading the model...")  # Debugging message
#     model = load_model('derma_disease_detection_VGG16_fine_tuned_colab_20241026_013722.h5')
#     st.write("Model loaded successfully!")  # Debugging message
#     return model

# # Dictionary to map predictions to diseases and prescriptions
# disease_prescription = {
#     0: ("Eczema", "Apply a moisturizing cream like CeraVe twice a day."),
#     1: ("Psoriasis", "Apply a topical corticosteroid and keep skin moisturized."),
#     2: ("Acne", "Use a salicylic acid cleanser and apply benzoyl peroxide cream.")
# }

# # Set up the Streamlit app
# st.title("Skin Disease Detection and Prescription")
# st.write("App is running!")  # Debugging message

# # Upload an image
# uploaded_file = st.file_uploader("Upload an image of the skin condition", type=["jpg", "jpeg", "png"])

# # Display the uploaded image
# if uploaded_file is not None:
#     st.write("Image uploaded!")  # Debugging message
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image to match the model's input size
#     img_array = img_to_array(image)
#     img_array = cv2.resize(img_array, (224, 224))  # VGG16 input size
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array /= 255.0  # Normalize the image

#     # Load the model
#     model = load_vgg16_model()

#     # Make a prediction
#     if st.button("Predict"):
#         st.write("Predicting...")  # Debugging message
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions)

#         # Display the result
#         disease, prescription = disease_prescription.get(predicted_class, ("Unknown", "No prescription available"))
#         st.success(f"Prediction: {disease}")
#         st.info(f"Prescription: {prescription}")
# else:
#     st.write("Please upload an image to proceed.")



# minimal_test_app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

st.write("App loaded!")  # Debugging message

# Load the fine-tuned VGG16 model
@st.cache(allow_output_mutation=True)
def load_vgg16_model():
    st.write("Loading model...")  # Debugging message
    try:
        # model = load_model('derma_disease_detection_VGG16_fine_tuned_colab_20241026_013722.h5')
        model = load_model('derma_disease_detection_VGG16_fine_tuned_colab_finetuned_20241027_202654.h5')
        st.write("Model loaded successfully!")  # Debugging message
        return model
    except Exception as e:
        st.write(f"Error loading model: {e}")  # Debugging message

# Dictionary to map predictions to diseases and prescriptions
disease_prescription = {
    0: ("Melanoma", "Consult a dermatologist for a biopsy as it may require medical treatment."),
    1: ("Nevus", "No treatment is usually necessary, but monitor for changes in shape or color."),
    2: ("Seborrheic Keratosis", "These are generally benign and can be removed if bothersome.")
}

st.title("Skin Disease Detection and Prescription")
st.write("Title displayed")  # Debugging message

# Upload an image
uploaded_file = st.file_uploader("Upload an image of the skin condition", type=["jpg", "jpeg", "png"])
st.write("File uploader displayed")  # Debugging message

# Display the uploaded image
if uploaded_file is not None:
    st.write("Image uploaded!")  # Debugging message
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match the model's input size
    try:
        img_array = img_to_array(image)
        img_array = cv2.resize(img_array, (224, 224))  # VGG16 input size
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image
        st.write("Image preprocessing completed")  # Debugging message
    except Exception as e:
        st.write(f"Error processing image: {e}")

    # Load the model
    model = load_vgg16_model()

    # Make a prediction
    if model and st.button("Predict"):
        st.write("Predicting...")  # Debugging message
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            disease, prescription = disease_prescription.get(predicted_class, ("Unknown", "No prescription available"))
            st.success(f"Prediction: {disease}")
            st.info(f"Prescription: {prescription}")
        except Exception as e:
            st.write(f"Error during prediction: {e}")
else:
    st.write("Please upload an image to proceed.")
