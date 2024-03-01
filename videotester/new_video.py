import cv2
import os
from gradio_client import Client
import time

client = Client("https://tsujuifu-ml-mgie.hf.space/--replicas/idy5w/")

def extract_frames(video_path, output_folder, fps):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Delete existing files in the output folder
        files = [os.path.join(output_folder, f) for f in os.listdir(output_folder)]
        for file in files:
            os.remove(file)

    for i in range(0, frame_count, int(cap.get(cv2.CAP_PROP_FPS) / fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{output_folder}/frame_{i}.png"
            cv2.imwrite(frame_name, frame)

    cap.release()

def create_video(frames_folder, output_video_path, fps):
    frames = [f for f in os.listdir(frames_folder) if f.endswith('.png')]
    
    # Ensure that only files with the expected pattern are considered
    frames = [f for f in frames if f.startswith('frame_') and f.endswith('.png')]

    # Sort frames based on their numerical part
    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(frames)
    # Check if there are any frames to process
    if not frames:
        print("No frames found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, layers = frame.shape

    # Delete existing video file
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frames:
        img = cv2.imread(os.path.join(frames_folder, frame_file))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    
def process_frame(frame):
    # Convert frame to image.png (or any other image format supported by your API)
    cv2.imwrite("image.png", frame)

    while True:
        try:
            # Use Gradio API to edit the image
            result = client.predict(
                "image.png",  # filepath in 'Input Image' Image component
                "make this like a pencil sketch",  # str in 'Instruction' Textbox component
                13331,  # float in 'Seed' Number component
                7.5,  # float in 'Text CFG' Number component
                1.5,  # float in 'Image CFG' Number component
                api_name="/go_mgie"
            )
            print("frame done")
            break
        except ValueError as e:
            error_message = str(e)
            if "exceeded your GPU quota" in error_message:
                print(error_message)
                print(f"API request exceeded GPU quota. Waiting for a minute and then retrying.")
                time.sleep(60)  # Wait for a minute before retrying
            else:
                raise  # Re-raise the exception if it's not related to GPU quota

    # Return the edited frame
    edited_frame_path, _ = result
    edited_frame = cv2.imread(edited_frame_path)
    return edited_frame
    
    
    
    
    
video_path = "carshort.mp4"
output_folder = "./Frames/"
edited_folder = "./EditedFrames/"
output_video_path = "EditedVideo.mp4"
target_fps = 5

if not os.path.exists(edited_folder):
    os.makedirs(edited_folder)
else:
    # Delete existing files in the output folder
    files = [os.path.join(edited_folder, f) for f in os.listdir(edited_folder)]
    for file in files:
        os.remove(file)

extract_frames(video_path, output_folder, target_fps)

# Using os.listdir() to get a list of all files in the folder
files = os.listdir(output_folder)


        
# Iterate through the files
for file in files:
    # Check if the item is a file (not a subfolder)
    image_path = os.path.join(output_folder, file)
    edited_frame = process_frame(cv2.imread(image_path))
    
    output_path = os.path.join(edited_folder, f"{file}")
    cv2.imwrite(output_path, edited_frame)
    
create_video(edited_folder, output_video_path, target_fps)
    
