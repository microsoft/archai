import numpy as np
from scipy import misc
from skimage import feature
import matplotlib.pyplot as plt
from PIL import Image





def main():
    f = misc.face()

    # resize to 16x16 to test hog at that scale
    f = np.array(Image.fromarray(f).resize((16,16)))
    
    fig = plt.imshow(f)

    hog_feat, hog_vis = feature.hog(f, pixels_per_cell = (4, 4), visualize=True)

    fig_hog = plt.imshow(hog_vis)
    
    plt.show()
    










if __name__ == '__main__':
    main()