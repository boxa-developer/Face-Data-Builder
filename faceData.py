import dlib
import numpy as np
import timeit
from scipy.spatial import distance
import os

class FaceCompare:
    def __init__(self):
        super(FaceCompare, self).__init__()
        self.init_settings()

    def init_settings(self):
        predictor_file_path = 'models/shape_predictor_68_face_landmarks.dat'
        recognition_file_path = 'models/dlib_face_recognition_resnet_model_v1.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_file_path)
        facerec = dlib.face_recognition_model_v1(recognition_file_path)
        self.detector, self.predictor, self.facerec = detector, predictor, facerec

    def computeDescriptor(self, filename):
        img = dlib.load_rgb_image(filename)
        dets = self.detector(img, 1)

        shape = self.predictor(img, dets[0])
        stop = timeit.default_timer()

        faces = dlib.full_object_detections()

        for detection in dets:
            faces.append(self.predictor(img, detection))
        stop = timeit.default_timer()

        face_ds = []
        for i in range(len(faces)):
            tmpArr = self.toNumpyArray(self.facerec.compute_face_descriptor(img, faces[i]))
            face_ds.append(tmpArr)
        return face_ds

    def toNumpyArray(self, vector):
        array = np.zeros(shape=128)
        for i in range(0, len(vector)):
            array[i] = vector[i]
        return array

    def computeSimilarity(self, x1, x2):
        x = distance.euclidean(x1, x2)
        return 100 * round(1 / (1 + x), 2)

s = timeit.default_timer()
obj = FaceCompare()
e = timeit.default_timer()
scanObj = os.scandir('./faces')
print(f'Faces Directory Loaded in {e-s}')
s = timeit.default_timer()
for item in scanObj:
    fname = os.path.join('faceData',item.name.split('.')[:-1][0]+'.npy')
    farr  = obj.computeDescriptor(os.path.join('faces', item.name))
    start = timeit.default_timer()
    np.save(fname, farr)
    stop = timeit.default_timer()
    print(f'Generated {fname} in {stop-start}')
e = timeit.default_timer()
print(f'Overall Time: {e-s}')

# np.save('f1.npy',obj.computeDescriptor('./faces/f1.jpg'))
# arr = np.load('f1.npy')



