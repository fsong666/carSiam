from os.path import join, isdir
from os import listdir, mkdir
import cv2
from ods.panoramaToCube import convertToCubes
from concurrent import futures
import sys
import time


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def gen(src, gen_path):
    imgIn = cv2.imread(src)
    imgOut = convertToCubes(imgIn)
    cv2.imwrite(join(gen_path, src.split('/')[-1]), imgOut)


def main(num_threads=4):
    gen_path = './cube_test'
    if not isdir(gen_path): mkdir(gen_path)
    src_path = './low_right_street'
    frames = sorted(listdir(src_path))
    n_frames = len(frames)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(gen, join(src_path, frame), gen_path) for frame in frames]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, n_frames, prefix='street', suffix='Done ', barLength=40)

    # img = join(src_path, frames[4])
    # gen(img, gen_path)
    #
    # img = cv2.imread(img)
    # cv2.imshow('img', img)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
