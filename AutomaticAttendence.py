import face_recognition
import cv2
import numpy as np
import csv
import glob
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import os

def load_known_faces():
    known_face_encoding = []
    known_face_name = []
    known_face_email = {}  

    
    for image_path in glob.glob(" paste path where your images are with images name as your real name "):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        
        if len(face_encodings) == 0:
            print("No face detected in", image_path)
            continue  
        
        encoding = face_encodings[0]
        known_face_encoding.append(encoding)
        name = image_path.split("\\")[-1].split(".")[0]
        known_face_name.append(name)
        known_face_email[name] = ""  

    
    if os.path.exists("known_face_email.json"):
        with open("known_face_email.json", "r") as file:
            known_face_email = json.load(file)

    return known_face_encoding, known_face_name, known_face_email

def save_known_faces_email(known_face_email):
    with open("known_face_email.json", "w") as file:
        json.dump(known_face_email, file)

def record_attendance(name, lnwriter, attendance_data, known_face_email):
    current_time = datetime.now().strftime("%H-%M-%S")
    lnwriter.writerow([name, current_time])
    print(f"Attendance recorded for: {name} at {current_time}")
    attendance_data[name] = attendance_data.get(name, 0) + 1  
    if known_face_email[name]:
        send_email_notification(f"Attendance recorded for: {name} at {current_time}", known_face_email[name])

def generate_attendance_summary(attendance_data):
    summary = "Attendance Summary:\n"
    for name, count in attendance_data.items():
        summary += f"{name}: {count} times present\n"
    return summary

def send_email_notification(body, receiver_email):
   
    sender_email = "your email"
    password = "your app password"

   
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Attendance for the day"

    
    message.attach(MIMEText(body, "plain"))

   
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)

def add_new_user(known_face_encoding, known_face_name, known_face_email):
    name = input("Enter the name of the new user: ")
    image_path = input("Enter the path to the image of the new user: ").strip('"')
    email = input(f"Enter the email address for {name}: ")
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encoding.append(encoding)
        known_face_name.append(name)
        known_face_email[name] = {"image_path": image_path, "email": email}  
        print("New user", name, "added successfully.")
    except Exception as e:
        print("Error:", e)
        add_new_user(known_face_encoding, known_face_name, known_face_email)  



def main():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Failed to open video capture device.")
        return

    known_face_encoding, known_face_name, known_face_email = load_known_faces()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    f = open(current_date + '.csv', 'w+', newline='')
    lnwriter = csv.writer(f)

    attendance_data = {}  

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            email = ""

            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_name[best_match_index]
                known_face_email[name] = email
                if name not in attendance_data:  
                    face_names.append(name)
                    record_attendance(name, lnwriter, attendance_data, known_face_email)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.imshow("attendance system", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()  

 
    summary = generate_attendance_summary(attendance_data)
    print(summary)

    
    for name, email in known_face_email.items():
        if email:
            send_email_notification(summary, email)  

   
    while True:
        add_user_choice = input("Do you want to add another new user? (yes/no): ").lower()
        if add_user_choice == "yes":
            add_new_user(known_face_encoding, known_face_name, known_face_email)
        elif add_user_choice == "no":
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    main()


