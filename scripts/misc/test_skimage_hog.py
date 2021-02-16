import numpy as np
from scipy import misc
from skimage import feature
import matplotlib.pyplot as plt





def main():
    f = misc.face()
    
    fig = plt.imshow(f)

    hog_feat, hog_vis = feature.hog(f, visualize=True)

    fig_hog = plt.imshow(hog_vis)
    
    plt.show()
    










if __name__ == '__main__':
    main()