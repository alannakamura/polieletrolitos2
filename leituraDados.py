import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('res.png')
plt.imshow(img)
plt.axis('off')   # remove os eixos
plt.show()