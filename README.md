# 🩺 Skin Disease Detection Web App

A simple web application built with **Streamlit** that uses a fine-tuned **VGG16** model to detect common skin conditions from uploaded images. It provides a prediction and a general suggestion based on the detected condition.

---

## 🚀 Features

-   🖼️ **Image Upload**: Allows users to upload images of skin conditions (JPG, JPEG, PNG).
-   🧠 **AI-Powered Prediction**: Uses a pre-trained VGG16 model to classify the image into one of the supported skin conditions.
-   📝 **Basic Suggestions**: Provides a general prescription or advice for the detected condition.
-   Supported conditions:
    -   Melanoma
    -   Nevus
    -   Seborrheic Keratosis

---

## 🛠️ Tech Stack

-   **Streamlit**: For creating the web application interface.
-   **TensorFlow/Keras**: For loading and using the deep learning model.
-   **Pillow (PIL)**: For image manipulation.
-   **OpenCV (cv2)**: For image preprocessing (resizing).
-   **NumPy**: For numerical operations.

---

## ⚙️ Setup and Installation

1.  **Clone the Repository (or download the files)**:
    If this project were in a repository, you'd clone it. For now, ensure you have `derma_web_app.py` and the model file in the same directory.

2.  **Model File**:
    You need the pre-trained model file: `derma_disease_detection_VGG16_fine_tuned_colab_finetuned_20241027_202654.h5`.
    Place this file in the same directory as `derma_web_app.py`.

3.  **Install Dependencies**:
    Create a virtual environment (recommended) and install the required Python packages. You can create a `requirements.txt` file with the following content:

    ```txt
    streamlit
    tensorflow
    numpy
    Pillow
    opencv-python
    ```

    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Running the Application

1.  Navigate to the directory containing `derma_web_app.py` and the model file.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run derma_web_app.py
    ```
3.  The application will open in your web browser.

---

## 📋 How to Use

1.  **Open the App**: Once the app is running, it will display the title "Skin Disease Detection and Prescription."
2.  **Upload Image**: Click on the "Upload an image of the skin condition" button and select an image file (JPG, JPEG, or PNG).
3.  **View Image**: The uploaded image will be displayed.
4.  **Predict**: Click the "Predict" button.
5.  **Get Results**: The application will show the predicted skin condition and a corresponding general prescription or advice.

---

## ⚠️ Disclaimer

This application is for informational purposes only and does not substitute professional medical advice. Always consult with a qualified dermatologist or healthcare provider for any medical concerns or before making any decisions related to your health.
