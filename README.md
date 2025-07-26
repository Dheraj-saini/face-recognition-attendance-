import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Directory for storing user images
IMAGE_DIR = 'images'

# CSV file for attendance logging
ATTENDANCE_FILE = 'attendance.csv'

def capture_image(user_name):
    """Capture and store user image."""
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Image", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:  # SPACE pressed
            img_name = f"{IMAGE_DIR}/{user_name}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            break

    cam.release()
    cv2.destroyAllWindows()

def encode_images():
    """Load and encode images from the directory."""
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(IMAGE_DIR):
        if file_name.endswith('.png'):
            image_path = os.path.join(IMAGE_DIR, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(file_name.split('.')[0])

    return known_face_encodings, known_face_names

def mark_attendance(name):
    """Mark attendance in the CSV file."""
    with open(ATTENDANCE_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, date_time])
        print(f"Attendance marked for {name} at {date_time}")

def recognize_faces():
    """Recognize faces and mark attendance."""
    known_face_encodings, known_face_names = encode_images()
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            mark_attendance(name)

        cv2.imshow("Recognize Faces", frame)
        if cv2.waitKey(1) % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

def delete_user(user_name):
    """Delete a registered user."""
    image_path = os.path.join(IMAGE_DIR, f"{user_name}.png")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted {user_name}'s image.")
    else:
        print(f"No image found for {user_name}.")

def main():
    """Main function to run the application."""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    while True:
        print("\n1. Capture Image")
        print("2. Recognize Faces")
        print("3. Delete User")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            user_name = input("Enter user name: ")
            capture_image(user_name)
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            user_name = input("Enter user name to delete: ")
            delete_user(user_name)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
