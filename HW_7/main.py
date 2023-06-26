import cv2
from convertToPolarImage import convertToPolarImage
from matplotlib import pyplot as plt


image_path = "/Users/marynavek/Projects/ComputerVision/structure.jpg"
        
image = cv2.imread(image_path, 0)
image = cv2.resize(image, (256,256))

polarImage, ptSettings = convertToPolarImage(image, center=[30, 30])
plt.imshow(polarImage.T, origin='lower', cmap='gray')
plt.show()