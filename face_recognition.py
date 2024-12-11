import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
import streamlit as st
import pickle  # For saving/loading the face database

# Initialize device and models outside of functions to avoid re-initialization
device = torch.device('cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize an empty database for storing known faces.
face_database = {}

def extract_face_embeddings(image, boxes):
    """Extract face embeddings from an image and bounding boxes."""
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)
        faces.append(face)

    if len(faces) > 0:
        faces = torch.cat(faces)
        embeddings = model(faces).detach().cpu().numpy()
        return embeddings

    return None

def add_faces_from_folder(folder_path):
    """Add all faces from a folder to the database using folder names as labels."""
    global face_database
    face_database.clear()  # Clear previous data before training anew
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for image_name in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    boxes,_= mtcnn.detect(img_rgb)

                    if boxes is not None:
                        embeddings = extract_face_embeddings(img_rgb, boxes)
                        if embeddings is not None:
                            if label not in face_database:
                                face_database[label] = []
                            face_database[label].append(embeddings[0])

def save_database(filename='face_database.pkl'):
    """Save the face database to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(face_database, f)

def load_database(filename='face_database.pkl'):
    """Load the face database from a file."""
    global face_database
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            face_database = pickle.load(f)

def detect_faces(image):
    """Detect faces in an image and draw bounding boxes with labels."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes,_= mtcnn.detect(img_rgb)

    if boxes is not None:
        embeddings=extract_face_embeddings(img_rgb ,boxes)

        for box ,face_embedding in zip(boxes ,embeddings):
            x1,y1,x2,y2=map(int ,box)
            cv2.rectangle(image,(x1,y1),(x2,y2),(0 ,0 ,255), 2)

            min_distance=float('inf')
            second_min_distance=float('inf')
            best_match="Unknown"

            for name ,known_embeddings in face_database.items():
                distances=[np.linalg.norm(face_embedding-known_emb) for known_emb in known_embeddings]
                current_min_distance=min(distances)

                if current_min_distance<min_distance:
                    second_min_distance=min_distance
                    min_distance=current_min_distance
                    best_match=name
                elif current_min_distance<second_min_distance:
                    second_min_distance=current_min_distance

            if min_distance < 0.9 and (second_min_distance - min_distance) > 0.1:  # Gap of 0.1
                label = best_match
            else:
                label = "Unknown"
            face_width = x2 - x1
            
            # Set font scale based on face width (adjust these factors as needed)
            min_font_scale = 0.75  # Minimum font scale to ensure readability
            max_font_scale = 1.5   # Maximum font scale to prevent overly large text
            
            # Determine font scale based on whether the label is 'Unknown'
            if label == "Unknown":
                font_scale = min_font_scale * 1.0  # Decrease size for 'Unknown'
            else:
                font_scale = max(min_font_scale, min(face_width / 100, max_font_scale))

            # Calculate text size and position
            font_thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

            # Draw a background rectangle for better visibility (optional)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 0, 255), -1)  # Red background

            # Put text above the bounding box without bold formatting
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    return image

def main():
    st.title("Real Time Face Recognition App")
    
    option = st.sidebar.selectbox("Choose an option", ["Train", "Test"])

    # Load existing database at start only for testing phase
    if option == "Test":
        load_database()

    if option == "Train":
        st.subheader("Training Phase")
        folder_path = st.text_input("Enter path to images folder:", 'C:\Users\Calibrage27\PycharmProjects\Facial Recognition\ImagesAttendance')
        if st.button("Train Model"):
            if folder_path:
                add_faces_from_folder(folder_path)  # Load images and labels from folders
                save_database()  # Save the trained model's database to a file
                st.success("Model trained successfully!")
            else:
                st.error("Please enter a valid folder path.")

    elif option == "Test":
        st.subheader("Testing Phase")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_img = detect_faces(img)  # Detect faces in the uploaded image

            # Display the output image
            output_image_path='output_image.jpg'
            cv2.imwrite(output_image_path, processed_img)
            st.image(output_image_path, caption='Processed Image', use_column_width=True)  

if __name__ == "__main__":
    main()
