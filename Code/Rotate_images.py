import cv2, os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
	os.chdir(script_dir)

def flip_images():
	gest_folder = "gestures"
	if not os.path.exists(gest_folder):
		print("No gestures folder found.")
		return
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		for i in range(1200):
			path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+1+1200)+".jpg"
			if os.path.exists(path):
				print(path)
				img = cv2.imread(path, 0)
				if img is not None:
					img = cv2.flip(img, 1)
					cv2.imwrite(new_path, img)

if __name__ == '__main__':
	flip_images()
