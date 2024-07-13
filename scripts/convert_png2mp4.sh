ffmpeg -y -f image2 -r 30 -pattern_type glob -i 'frames/frame_????.png' -crf 22 video.mp4
