import cv2, os, random
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
	os.chdir(script_dir)

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	if img is None:
		return (50, 50)
	return img.shape

def display():
	if not os.path.exists('gestures') or not os.listdir('gestures'):
		print("No gestures folder or images found to display.")
		return
	gestures = os.listdir('gestures/')
	gestures.sort(key = int)
	begin_index = 0
	end_index = 5
	image_x, image_y = get_image_size()

	if len(gestures)%5 != 0:
		rows = int(len(gestures)/5)+1
	else:
		rows = int(len(gestures)/5)

	full_img = None
	for i in range(rows):
		col_img = None
		for j in range(begin_index, end_index):
			img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
			img = cv2.imread(img_path, 0)
			if np.any(img == None):
				img = np.zeros((image_y, image_x), dtype = np.uint8)
			if np.any(col_img == None):
				col_img = img
			else:
				col_img = np.hstack((col_img, img))

		begin_index += 5
		end_index += 5
		if np.any(full_img == None):
			full_img = col_img
		else:
			full_img = np.vstack((full_img, col_img))

	cv2.imshow("gestures", full_img)
	cv2.imwrite('full_img.jpg', full_img)
	cv2.waitKey(0)

if __name__ == '__main__':
	display()
