"""
    Compute depth maps for images in the input folder.
"""

import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')


def _minify(basedir, factors=[], resolutions=[]):
    '''
        Minify the images to small resolution for training
    '''

    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output
    import glob

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir


    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])

        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        print(ext)
        # sys.exit()
        img_path_list = glob.glob(os.path.join(imgdir, '*.%s' % ext))

        for img_path in img_path_list:
            save_path = img_path.replace('.jpg', '.png')
            img = cv2.imread(img_path)

            print(img.shape, r)
            # sys.exit()

            cv2.imwrite(save_path,
                        cv2.resize(img,
                                   (r[1], r[0]),
                                   interpolation=cv2.INTER_AREA))

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def run(basedir, resize_height=288):
    """Run MonoDepthNN to compute depth maps.
    """
    print("initialize")

    img0 = [os.path.join(basedir, 'images', f) \
            for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = cv2.imread(img0).shape
    height = resize_height
    factor = sh[0] / float(height)
    width = int(round(sh[1] / factor))
    _minify(basedir, resolutions=[[height, width]])

    print("finished")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help='COLMAP Directory')
    parser.add_argument("--resize_height", type=int, default=288,
                        help='resized image height for training \
                        (width will be resized based on original aspect ratio)')

    args = parser.parse_args()
    BASE_DIR = args.data_path

    # compute depth maps
    run(BASE_DIR, args.resize_height)


