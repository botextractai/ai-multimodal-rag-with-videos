import cv2
import json
import os
import webvtt
import whisper
from moviepy.editor import VideoFileClip
from os import path as osp
from pathlib import Path
from urllib.request import urlretrieve
from utils import download_video, get_transcript_vtt
from utils import lvlm_inference, encode_image
from utils import maintain_aspect_ratio_resize
from utils import str2time


# DOWNLOAD VIDEOS
# ===============

# first video's url
vid1_url = "https://www.youtube.com/watch?v=7Hcg-rLYwdM"
# download Youtube video to ./shared_data/videos/video1
vid1_dir = "./shared_data/videos/video1"
vid1_filepath = download_video(vid1_url, vid1_dir)
# download Youtube video's subtitle to ./shared_data/videos/video1
vid1_transcript_filepath = get_transcript_vtt(vid1_url, vid1_dir)

# second video's url
vid2_url = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/010/a07/010a074acb1975c4d6d6e43c1faeb8.mp4"
vid2_dir = "./shared_data/videos/video2"
vid2_name = "toddler_in_playground.mp4"

# create folder to which video2 will be downloaded 
Path(vid2_dir).mkdir(parents=True, exist_ok=True)
vid2_filepath = urlretrieve(
                        vid2_url, 
                        osp.join(vid2_dir, vid2_name).replace("\\","/")
                    )[0]


# CASE 1 - VIDEO AND ITS TRANSCRIPT ARE AVAILABLE
# ===============================================

#   receives as input a video and its transcript
#   extracts and saves frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata(
        path_to_video, 
        path_to_transcript, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    trans = webvtt.read(path_to_transcript)
    
    # iterate transcript file
    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        # get the start time and end time in seconds
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        # get the time in ms exactly 
        # in the middle of start time and end time
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", ' ')
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            ).replace("\\","/")
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = osp.join(path_to_save_metadatas, 'metadatas.json').replace("\\","/")
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

# output paths to save extracted frames and their metadata 
extracted_frames_path = osp.join(vid1_dir, 'extracted_frame').replace("\\","/")
metadatas_path = vid1_dir

# create these output folders if not existing
Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

# call the function to extract frames and metadatas
metadatas = extract_and_save_frames_and_metadata(
                vid1_filepath, 
                vid1_transcript_filepath,
                extracted_frames_path,
                metadatas_path,
            )


# CASE 2 - VIDEO WITHOUT AVAILABLE TRANSCRIPT
# ===========================================

path_to_video_no_transcript = vid1_filepath

# declare where to save .mp3 audio
path_to_extracted_audio_file = os.path.join(vid1_dir, 'audio.mp3').replace("\\","/")

# extract mp3 audio file from mp4 video video file
clip = VideoFileClip(path_to_video_no_transcript)
clip.audio.write_audiofile(path_to_extracted_audio_file)

model = whisper.load_model("small")
options = dict(task="translate", best_of=1, language='en')
results = model.transcribe(path_to_extracted_audio_file, **options)

from utils import getSubs
vtt = getSubs(results["segments"], "vtt")

# path to save generated transcript of video1
path_to_generated_trans = osp.join(vid1_dir, 'generated_video1.vtt').replace("\\","/")
# write transcription to file
with open(path_to_generated_trans, 'w') as f:
    f.write(vtt)


# CASE 3 - VIDEO WITHOUT LANGUAGE (SILENT OR JUST MUSIC)
# ======================================================

lvlm_prompt = "Can you describe the image?"

# function extract_and_save_frames_and_metadata_with_fps
# receives as input a video 
# does extracting and saving frames and their metadatas
# returns the extracted metadatas
def extract_and_save_frames_and_metadata_with_fps(
        path_to_video,  
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    
    # get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    curr_frame = 0
    idx = -1
    while(True):
        # iterate all frames
        ret, frame = video.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            idx = idx + 1
        
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames, 
                            img_fname
                        ).replace("\\","/")
            cv2.imwrite(img_fpath, image)

            # generate caption using lvlm_inference
            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)
                
            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1
        
    # save metadata of all extracted frames
    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json').replace("\\","/")
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

# paths to save extracted frames and metadata (their transcripts)
extracted_frames_path = osp.join(vid2_dir, 'extracted_frame').replace("\\","/")
metadatas_path = vid2_dir

# create these output folders if not existing
Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

# call the function to extract frames and metadatas
metadatas = extract_and_save_frames_and_metadata_with_fps(
                vid2_filepath, 
                extracted_frames_path,
                metadatas_path,
                num_of_extracted_frames_per_second=0.1
            )
