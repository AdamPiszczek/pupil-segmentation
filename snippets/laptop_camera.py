import cv2
import time
import os
from tkinter import Tk, Canvas

# Step 1: Create a full-screen black window
def create_black_screen():
    root = Tk()
    root.attributes('-fullscreen', True)
    canvas = Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), bg='black')
    canvas.pack()
    return root, canvas

# Step 2: Capture video according to the specified sequence
def capture_video(output_folder, canvas):
    cap = cv2.VideoCapture(0)
    frames = []
    start_time = time.time()
    video_start_time = start_time + 5  # Start video at t=5s
    flash_start_time = video_start_time + 1  # Flash starts at t=6s
    flash_end_time = flash_start_time + 1  # Flash ends at t=7s
    video_end_time = flash_end_time + 2  # End video at t=9s

    while time.time() < video_end_time:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time >= video_start_time:
            frames.append(frame)

        if flash_start_time <= current_time < flash_end_time:
            canvas.config(bg='white')
        else:
            canvas.config(bg='black')
        canvas.update()

    cap.release()

    # Save frames after capturing
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_folder, f"frame_{i}.jpg"), frame)

    cv2.destroyAllWindows()

# Main function
def main():
    project_directory = os.getcwd()
    output_folder = os.path.join(project_directory, "output")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    root, canvas = create_black_screen()
    canvas.config(bg='black')
    canvas.update()
    # Wait 5 seconds before starting the video
    time.sleep(5)
    capture_video(output_folder, canvas)
    root.destroy()

if __name__ == "__main__":
    main()
