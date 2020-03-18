from utils import *
np.seterr(divide='ignore', invalid='ignore')

EPSILON = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='dataset to process (obama or trump)')
parser.add_argument('--trim')
parser.add_argument('--extract_images')
parser.add_argument('--extract_audio')
parser.add_argument('--extract_image_kp')
parser.add_argument('--extract_pca')
parser.add_argument('--extract_audio_kp')


if __name__ == '__main__':

	args = parser.parse_args()
	dataset = args.dataset

	################################################################################
	if args.trim:

		inputFolder = dataset + '/videos/'
		outputFolder = dataset + '/trimmed_videos/'

		video_width = 456

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		for i in range(1, 11):
			num = str(i).zfill(5)
			captions_filename = '{}/captions/{}.en.vtt'.format(dataset, num)
			inputfilename = inputFolder + num + '.mp4 '
			captions = WebVTT().read(captions_filename)

			print('Total Length of captions is', len(captions))

			for idx, caption in enumerate(captions):
				outputfilename = outputFolder + num + '-' + str(idx).rjust(3, '0') + '.mp4'
				ss = caption.start
				end = caption.end
				t = int(round(get_sec(end) - get_sec(ss)))
				print('index: ', idx, 'Text: ', caption.text, 'Time: ', t)
				cmd = 'ffmpeg -i {} -vf scale={}:256 -ss {} -t {} -acodec copy {}'.format(inputfilename, video_width, ss, t, outputfilename)
				subprocess.call(cmd, shell=True)

	################################################################################
	if args.extract_images:

		inputFolder = dataset + '/trimmed_videos/'
		outputFolder = dataset + '/images/'

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		filelist = sorted(glob(inputFolder+'/*.mp4'))

		print('Length of filelist: ', len(filelist))

		for idx, filename in tqdm(enumerate(filelist)):
			num = filename[len(inputFolder):-len('.mp4')]
			print('Num: ', num)
			# Create this directory if it doesn't exist
			if not os.path.exists(outputFolder+num):
				# Create directory
				subprocess.call('mkdir -p ' + outputFolder+num, shell=True)

			# Create the images
			cmd = 'ffmpeg -i ' + filename + ' -vf scale=-1:256 '+ outputFolder + num + '/$filename%05d' + '.bmp'
			subprocess.call(cmd, shell=True)

			# Cropping
			imglist = sorted(glob( outputFolder + num + '/*.bmp'))

			for i in range(len(imglist)):
				img = cv2.imread(imglist[i])
				x = int(np.floor((img.shape[1]-256)/2))
				crop_img = img[0:256, x:x+256]
				cv2.imwrite( imglist[i][0:-len('.bmp')] + '.jpeg', crop_img)

			subprocess.call('rm -rf '+ outputFolder + num + '/*.bmp', shell=True)

	################################################################################
	if args.extract_audio:

		inputFolder = dataset + '/trimmed_videos/'
		outputFolder = dataset + '/audios/'

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		filelist = sorted(glob(inputFolder+'/*.mp4'))

		for file in filelist:
			cmd = 'ffmpeg -i ' + file + ' -ab 160k -ac 1 -ar 16000 -vn ' + outputFolder + file[len(inputFolder): -len('.mp4')] + '.wav'
			subprocess.call(cmd, shell=True)

	################################################################################
	if args.extract_image_kp:

		inputFolder = dataset + '/images/'
		outputFolder = dataset + '/image_kp_raw/'
		resumeFrom = 0

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		directories = sorted(glob(inputFolder+'*/'))

		d = {}

		for idx, directory in tqdm(enumerate(directories[resumeFrom:])):
			key = directory[len(inputFolder):-1]
			imglist = sorted(glob(directory+'*.jpeg'))
			big_list = []
			prev_store_list = [] # Alex: no face present in jpeg
			for file in tqdm(imglist):
				keypoints = get_facial_landmarks(file)
				if not keypoints.shape[0] == 1: # if there are some kp thn
					l = getKeypointFeatures(keypoints)
					unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
					kp_mouth = unit_kp[48:68]
					store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
					prev_store_list = store_list
					big_list.append(store_list)
				else:
					big_list.append(prev_store_list)
			d[key] = big_list

			saveFilename = outputFolder + 'kp' + str(idx+resumeFrom+1) + '.pickle'
			oldSaveFilename = outputFolder + 'kp' + str(idx+resumeFrom-2) + '.pickle'

			if not os.path.exists(saveFilename):
				with open(saveFilename, 'wb') as output_file:
					pkl.dump(d, output_file)
					print('Saved output for ', idx+resumeFrom+1, ' file.')
			else:
				# Resume
				with open(saveFilename, 'rb') as output_file:
					d = pkl.load(output_file)
					print('Loaded output for ', idx+resumeFrom+1, ' file.')

			# Keep removing stale versions of the files
			if os.path.exists(oldSaveFilename):
				cmd = 'rm -rf ' + oldSaveFilename
				subprocess.call(cmd, shell=True)

	################################################################################
	if args.extract_audio_kp:

		inputFolder = dataset + '/audios/'
		outputFolder = dataset + '/audio_kp/'
		resumeFrom = 0
		frame_rate = 5
		kp_type = 'mel'

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		filelist = sorted(glob(inputFolder+'*.wav'))

		d = {}

		for idx, file in enumerate(tqdm(filelist[resumeFrom:])):
			key = file[len(inputFolder):-len('.wav')]

			if kp_type == 'world':
				x, fs = sf.read(file)
				# 2-1 Without F0 refinement
				f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
								channels_in_octave=2,
								frame_period=frame_rate,
								speed=1.0)
				sp = pw.cheaptrick(x, f0, t, fs)
				ap = pw.d4c(x, f0, t, fs)
				features = np.hstack((f0.reshape((-1, 1)), np.hstack((sp, ap))))

			elif kp_type == 'mel':
				(rate, sig) = wav.read(file)
				features = logfbank(sig,rate)

			d[key] = features

			saveFilename = outputFolder + 'audio_kp' + str(idx+resumeFrom+1) + '_' + kp_type + '.pickle'
			oldSaveFilename = outputFolder + 'audio_kp' + str(idx+resumeFrom-2) + '_' + kp_type + '.pickle'

			if not os.path.exists(saveFilename):
				with open(saveFilename, 'wb') as output_file:
					pkl.dump(d, output_file)
					# print('Saved output for', idx+resumeFrom+1, 'file.')
			else:
				# Resume
				with open(saveFilename, 'rb') as output_file:
					d = pkl.load(output_file)
					print('Loaded output for ', idx+resumeFrom+1, ' file.')

			# Keep removing stale versions of the files
			if os.path.exists(oldSaveFilename):
				cmd = 'rm -rf ' + oldSaveFilename
				subprocess.call(cmd, shell=True)

		print('Saved Everything')

	################################################################################
	if args.extract_pca:

		inputFolder = dataset + '/image_kp_raw/'
		outputFolder = dataset + '/pca/'
		numOfFiles = 333 # First 5 obama videos
		# numOfFiles = 1414 # First obama 20 videos
		# numOfFiles = 250 # First 5 trump videos
		#numOfFiles = 137 # First 2 videos
		new_list = []

		filename = inputFolder + 'kp' + str(numOfFiles) + '.pickle'

		if not os.path.exists(outputFolder):
			# Create directory
			subprocess.call('mkdir -p ' + outputFolder, shell=True)

		if os.path.exists(filename):
			with open(filename, 'rb') as file:
				big_list = pkl.load(file)
			print('Keypoints file loaded')
		else:
			print('Input keypoints not found')
			sys.exit(0)

		print('Unwrapping all items from the big list')

		key_skipped = 0#
		frame_kp_skipped = 0#
		for key in tqdm(sorted(big_list.keys())):
			if len(big_list[key]) == 0:
				print('key:', key)
				key_skipped += 1
			for frame_kp in big_list[key]:
				if len(frame_kp) == 0:
					frame_kp_skipped += 1
					continue
				kp_mouth = frame_kp[0]
				x = kp_mouth[:, 0].reshape((1, -1))
				y = kp_mouth[:, 1].reshape((1, -1))
				X = np.hstack((x, y)).reshape((-1)).tolist()
				new_list.append(X)

		print('key_skipped:', key_skipped) #
		print('frame_kp_skipped:', frame_kp_skipped) #

		X = np.array(new_list)

		pca = PCA(n_components=8)
		pca.fit(X)
		with open(outputFolder + 'pca' + str(numOfFiles) + '.pickle', 'wb') as file:
			pkl.dump(pca, file)

		with open(outputFolder + 'explanation' + str(numOfFiles) + '.pickle', 'wb') as file:
			pkl.dump(pca.explained_variance_ratio_, file)

		print('Explanation for each dimension:', pca.explained_variance_ratio_)
		print('Total variance explained:', 100*sum(pca.explained_variance_ratio_))
		print('')
		print('Upsampling...')

		# Upsample the lip keypoints
		skipped = 0  # Alex: keep track of invalid frames
		upsampled_kp = {}
		for key in tqdm(sorted(big_list.keys())):
			# print('Key:', key)
			nFrames = len(big_list[key])
			if len(big_list[key]) == 0 or len(big_list[key][0]) == 0:
				skipped += 1
				continue
			factor = int(np.ceil(100/29.97))
			# Create the matrix
			new_unit_kp = np.zeros((int(factor*nFrames), big_list[key][0][0].shape[0], big_list[key][0][0].shape[1]))
			new_kp = np.zeros((int(factor*nFrames), big_list[key][0][-1].shape[0], big_list[key][0][-1].shape[1]))
			# print('Shape of new_unit_kp:', new_unit_kp.shape, 'new_kp:', new_kp.shape)

			for idx, frame in enumerate(big_list[key]):
				# Create two lists, one with original keypoints, other with unit keypoints
				new_kp[(idx*(factor)), :, :] = frame[-1]
				new_unit_kp[(idx*(factor)), :, :] = frame[0]

				if idx > 0:
					start = (idx-1)*factor + 1
					end = idx*factor
					for j in range(start, end):
						new_kp[j, :, :] = new_kp[start-1, :, :] + ((new_kp[end, :, :] - new_kp[start-1, :, :])*(np.float(j+1-start)/np.float(factor)))
						# print('')
						l = getKeypointFeatures(new_kp[j, :, :])
						# print('')
						new_unit_kp[j, :, :] = l[0][48:68, :]

			upsampled_kp[key] = new_unit_kp
		print('skipped:', skipped)

		# Use PCA to de-correlate the points
		d = {}
		keys = sorted(upsampled_kp.keys())
		for key in tqdm(keys):
			x = upsampled_kp[key][:, :, 0]
			y = upsampled_kp[key][:, :, 1]
			X = np.hstack((x, y))
			X_trans = pca.transform(X)
			d[key] = X_trans

		with open(outputFolder + 'pkp' + str(numOfFiles) + '.pickle', 'wb') as file:
			pkl.dump(d, file)
		print('Saved Everything')
