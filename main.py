import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([48, 48, 148])
    upper_bound = np.array([158, 255, 255])

    # Create a mask to extract blue parts of the image
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask=mask)

    # apply open and close morph operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    threshold = 1
    _, binary_img = cv2.threshold(res[:, :, 2], threshold, 255, cv2.THRESH_BINARY)

    total_sum = cv2.sumElems(binary_img)[0]

    return binary_img, total_sum

# Path to the image
# IMAGE_PATH = "/Users/user/Downloads/ambulance sirens/IMG_6685/"
IMAGE_PATH = "/Users/user/Downloads/ambulance sirens/IMG_6681/"

fps = 25
size = (600, 300)
# writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
sums = []
for frame_i in range(1, 560):
    frame_path = os.path.join(IMAGE_PATH, f'{frame_i}_0.jpg')
    frame = cv2.imread(frame_path)
    assert frame is not None

    # crop top 20% of the image
    top_k = int(frame.shape[0] * 0.2)
    frame = frame[:top_k, :, :]

    highlight, sub_sum = filter_color(frame)
    sums.append(sub_sum)

    # Concatenate the original and the highlighted image vertically
    highlight = cv2.cvtColor(highlight, cv2.COLOR_GRAY2BGR)
    image = np.concatenate((frame, highlight), axis=0)
    image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)
    # video_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # video_frame[:image.shape[0], :image.shape[1], :] = image
    # writer.write(video_frame)
    cv2.imshow('frame', image)
    if cv2.waitKey(1_000 // fps) & 0xFF == ord('q'):
        break

# show the plot
plt.plot(sums)
plt.show()

# writer.release()
cv2.destroyAllWindows()
