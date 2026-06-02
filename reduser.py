import os, cv2
from moviepy.editor import VideoFileClip

for i in os.listdir('stable_drive'):
	for data in VideoFileClip(f'stable_drive/{i}').iter_frames(fps = 2):
		cv2.imwrite('frame.jpg', data)
		break

	height, width, channels  = cv2.imread('frame.jpg').shape
	if height > width:
		os.remove(f'stable_drive/{i}')

os.remove(f'frame.jpg')