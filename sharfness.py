# # import cv2
# # import numpy as np

# # # Load two images
# # img1 = cv2.imread("785868.jpeg")
# # img2 = cv2.imread("b.jpg")

# # # Convert the images to grayscale
# # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # # Calculate the Laplacian for both images
# # laplacian1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
# # laplacian2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

# # # Compare the Laplacian values to determine which image is sharper
# # if laplacian1 > laplacian2:
# #     print("Image 1 is sharper.")
# # else:
# #     print("Image 2 is sharper.")

# import cv2
# import numpy as np

# # Load the image
# img = cv2.imread("785868.jpeg")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# var = np.var(laplacian)

# threshold = 100

# # Check if the variance is below the threshold
# if var < threshold:
#     print("The image is blurry.")
# else:
#     print("The image is not blurry.")

import cv2
import numpy as np

# Load the video capture
cap = cv2.VideoCapture(0)

# Define a threshold value to determine if the image is blurry
threshold = 100

# Loop through each frame of the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the variance of the Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var = np.var(laplacian)

        # Check if the variance is below the threshold
        if var < threshold:
            print("The frame is blurry.")
        else:
            print("The frame is not blurry.")

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check if the user presses 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
