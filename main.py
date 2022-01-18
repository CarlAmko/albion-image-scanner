import json

import cv2
import numpy as np
from matplotlib import pyplot as plt


def trim_to_content(_img):
    y, x = _img[:, :, 3].nonzero()  # get the nonzero alpha coordinates
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    return _img[miny:maxy, minx:maxx]


if __name__ == '__main__':
    img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
    img2 = img.copy()

    # Read template to search for
    tpl_file = '6_poison_pot.png'
    template = cv2.imread(tpl_file, cv2.IMREAD_UNCHANGED)

    # Trim image to content to improve matching
    template = trim_to_content(template)
    # cv2.cvtColor(template, cv2.RG)
    # cv2.imwrite('4_poison_pot.png', template)
    #
    # template = cv2.imread('4_poison_pot.png', cv2.IMREAD_COLOR)

    # Scale image down to improve matching
    template = cv2.resize(template, None, fx=0.35, fy=0.35)
    w, h = template.shape[:2]
    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

    results = {}
    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        results[meth] = {'min': min_val, 'max': max_val}

        plt.subplot(121), plt.imshow(template)
        plt.title('Search Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

    with open(f'results/results_{tpl_file.split(".")[0]}.json', 'w') as file:
        file.write(json.dumps(results))
