import os
import subprocess

def video_to_mp3(data_dir):
    video_dir = os.path.join(data_dir, "urfunny2_videos")
    vid_names = os.listdir(video_dir)
    audio_dir = os.path.join(data_dir, "acoustic")
    for vid in vid_names:
        vid_name = os.path.splitext(vid)[0]
        if not os.path.exists(os.path.join(audio_dir, vid_name + ".mp3")):
            vid_path = os.path.join(video_dir, vid)
            output = os.path.join(audio_dir, vid_name + ".mp3")
            subprocess.call('ffmpeg -i {video} {out_name}'.format(video=vid_path, out_name=output), shell=True)

    print("Finished extracting frames.")

def mp3_to_wav(data_dir):
    mp3_dir = os.path.join(data_dir, "acoustic")
    mp3_names = os.listdir(mp3_dir)
    wav_dir = os.path.join(data_dir, "acoustic_wav")
    for f in mp3_names:
        name = os.path.splitext(f)[0]
        if not os.path.exists(os.path.join(wav_dir, name + ".wav")):
            mp3_file_path = os.path.join(mp3_dir, f)
            output = os.path.join(wav_dir, name + ".wav")
            subprocess.call('ffmpeg -i {video} -acodec pcm_s16le -ac 1 -ar 22050 {out_name}'.format(video=mp3_file_path, out_name=output), shell=True)

if __name__ == "__main__":
    mp3_to_wav("/home/aobolens/urfunny")