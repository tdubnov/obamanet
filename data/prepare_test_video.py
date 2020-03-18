# Get keypoint data from the images and prepare for pix2pix
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='dataset to process (obama or trump)')
parser.add_argument('--jaw', help='whether use jaw keypoints')


if __name__ == '__main__':

    args = parser.parse_args()

    kp_str = 'jaw' if args.jaw else 'lip'

    inputFolder = args.dataset + '/images/'
    outputFolderKp = args.dataset + '/kp_{}/'.format(kp_str)
    saveFilename = outputFolderKp + 'kp_test.pickle'
    outputFolderForpix2pix = args.dataset + '/pix2pix_input_{}/'.format(kp_str)
    inputToA2KeyModel = args.dataset + '/kp_{}/images/'.format(kp_str)

    if not os.path.exists(outputFolderForpix2pix):
    	# Create directory
    	subprocess.call('mkdir -p ' + outputFolderForpix2pix, shell=True)
    if not os.path.exists(outputFolderKp):
    	# Create directory
    	subprocess.call('mkdir -p ' + outputFolderKp, shell=True)
    if not os.path.exists(inputToA2KeyModel):
    	# Create directory
    	subprocess.call('mkdir -p ' + inputToA2KeyModel, shell=True)

    searchNames = inputFolder + '*/*' + '.jpeg'
    filenames = sorted(glob(searchNames, recursive=True))

    d = []
    for file in tqdm(filenames):
        #print(file)
        try:
            img = cv2.imread(file)
            x = int(np.floor((img.shape[1]-256)/2))
            # Crop to a square image
            crop_img = img
            outputName = file

            # extract the keypoints
            keypoints = get_facial_landmarks(outputName)
            l = getKeypointFeatures(keypoints)
            unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
            kp_mouth = unit_kp[48:68]
            kp_jaw = unit_kp[0:17]
            store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
            d.append(store_list)

            # create a patch based on the tilt, mean and the size of face
            mean_x, mean_y = int(mean[0]), int(mean[1])
            if args.jaw:
                size = int(N/10)
            else:
                size = int(N/15)
            aspect_ratio_mouth = 1.8
            # print('mean (y, x):', mean_y, mean_x, 'size:', size)
            patch_img = crop_img.copy()
            if args.jaw:
                patch_img[mean_y-int(0.5*size): mean_y+int(1.2*size), mean_x-int(aspect_ratio_mouth*size):
                          mean_x+int(0.85*aspect_ratio_mouth*size)] = 0
            else:
                patch_img[mean_y-size: mean_y+size, mean_x-int(aspect_ratio_mouth*size):
                          mean_x+int(aspect_ratio_mouth*size)] = 0
            cv2.imwrite(inputToA2KeyModel + file[len(inputFolder):-len('/00001.jpeg')]+file[len(inputFolder)+len('00001-000/'):-len('.jpeg')] + '.png', patch_img)
            drawLips(keypoints, patch_img)
            if args.jaw:
                drawJaws(keypoints, patch_img)

            # Slap the other original image onto this
            patch_img = np.hstack((patch_img, crop_img))
            outputNamePatch = outputFolderForpix2pix + file[len(inputFolder):-len('/00001.jpeg')]+file[len(inputFolder)+len('00001-000/'):-len('.jpeg')] + '.png'
            cv2.imwrite(outputNamePatch, patch_img)
        except:
            pass
    # save the extracted keypoints
    with open(saveFilename, "wb") as output_file:
    	pkl.dump(d, output_file)
