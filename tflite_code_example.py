from turtle import right
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
VER = 'ver1'
TEMPLATE_LANDMARK_INDEX_22 = [
    33, 160, 158, 133, 153, 144,   ## eye right
    362, 385, 387, 263, 373, 380,  ## eye left
    168, 197, 5, 4,                ## nose bridge
    2,                             ## nose bottom
    61, 291,
    148, 152, 377,                 ## jaws
]

TEMPLATE_LANDMARK_INDEX_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

LEFT_LADNMARK_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_LADNMARK_IDX = [33, 160, 158, 133, 153, 144]

def get_roi(single_eye_landmarks, padding=1.5, ratio=0.6):
    np_landmarks = np.array(single_eye_landmarks)
    left, top = np.min(np_landmarks[:, 0]), np.min(np_landmarks[:, 1])
    right, bottom = np.max(np_landmarks[:, 0]), np.max(np_landmarks[:, 1])

    width = right-left
    height = bottom-top

    target_width = max(width, height/ratio) * (1+padding)
    target_height = target_width * ratio

    if target_width > width:
        left_padding = (target_width - width) / 2.0
        right_padding = (target_width - width) - left_padding
        left -= left_padding
        right += right_padding

    if target_height > height:
        bottom_padding = (target_height - height) / 2.0
        top_padding = (target_height - height) - bottom_padding
        top -= top_padding
        bottom += bottom_padding
    return [[left, top], [left, bottom], [right, top], [right, bottom]]

def get_single_eye_image(image, single_eye_landmarks, image_shape=(60, 36)):
    roi = get_roi(single_eye_landmarks)
    dst = [[0, 0], [0, image_shape[1]], [image_shape[0], 0], [image_shape[0], image_shape[1]]]
    M = cv2.getPerspectiveTransform(np.array(roi, dtype=np.float32), np.array(dst, dtype=np.float32))
    eye_img = cv2.warpPerspective(image, M, image_shape)
    return eye_img

def rotate_point(point, center, angle):
    # cos -sin
    # sin cos
    x, y = point[0] - center[0], point[1] - center[1]
    rot_x = math.cos(angle)*x - math.sin(angle)*y + center[0]
    rot_y = math.sin(angle)*x + math.cos(angle)*y + center[1]
    return [rot_x, rot_y]

def get_eye_images(image, eye_landmarks):
    r_landmarks, l_landmarks = eye_landmarks[:6], eye_landmarks[6:]
    r_center, l_center = np.mean(r_landmarks, axis=0), np.mean(l_landmarks, axis=0)
    angle = math.atan((r_center[1]-l_center[1])/(r_center[0]-l_center[0]))
    M = cv2.getRotationMatrix2D((r_center+l_center)/2, angle * 180 / math.pi, 1.0)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    eye_landmarks = list(map(lambda p: rotate_point(p, (r_center+l_center)/2, -angle), eye_landmarks))
    return cv2.flip(get_single_eye_image(image, eye_landmarks[:6]), 1), get_single_eye_image(image, eye_landmarks[6:])

def build_model(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def model_run(model_interpreter, left_landmark, right_landmark, image):

    input_index = model_interpreter.get_input_details()[0]['index']
    output_index = model_interpreter.get_output_details()[0]['index']
    landmarks_list = right_landmark+ left_landmark

    r_img, l_img = get_eye_images(image, landmarks_list)

    r_img_input = tf.image.convert_image_dtype(r_img, dtype=tf.float32)
    l_img_input = tf.image.convert_image_dtype(l_img, dtype=tf.float32)

    x=tf.expand_dims(r_img_input, axis=0)
    x = tf.cast(x, dtype=tf.float32)

    model_interpreter.set_tensor(input_index,x)
    model_interpreter.invoke()
    R_pred=model_interpreter.get_tensor(output_index)

    x=tf.expand_dims(l_img_input, axis=0)
    x = tf.cast(x, dtype=tf.float32)

    model_interpreter.set_tensor(input_index,x)
    model_interpreter.invoke()
    L_pred=model_interpreter.get_tensor(output_index)
    
    if VER == 'ver1':
        return round(1 - R_pred[0][0], 3), round(1 - L_pred[0][0], 3), r_img, l_img
    if VER == 'ver2':
        return round(1 - R_pred[0][0], 3), round(1 - R_pred[0][1], 3), round(1 - R_pred[0][2], 3), round(1 - L_pred[0][0], 3), round(1 - L_pred[0][1], 3), round(1 - L_pred[0][2], 3), r_img, l_img

def detect_face(img):
    height = img.shape[0]
    width = img.shape[1]

    start_lst = []
    end_lst = []

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, model_selection=0) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return start_lst, end_lst
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            start_lst.append([int(bbox.xmin * width), int(bbox.ymin * height)])
            end_lst.append([int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)])

    return start_lst, end_lst

def detect_face_landmark(image):

    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        iter = 0
        left_landmark = [0,0,0,0,0,0]
        right_landmark = [0,0,0,0,0,0]

        
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                shape = image.shape 
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                if iter in TEMPLATE_LANDMARK_INDEX_68:

                    cv2.circle(image, (relative_x, relative_y), radius=1, color=(0, 0, 255), thickness=2)
                if iter in RIGHT_LADNMARK_IDX:
                    temp = RIGHT_LADNMARK_IDX.index(iter)
                    right_landmark[temp] = (relative_x, relative_y)
                if iter in LEFT_LADNMARK_IDX:
                    temp = LEFT_LADNMARK_IDX.index(iter)
                    left_landmark[temp] = (relative_x, relative_y)
                iter+=1
        return image, left_landmark, right_landmark

if __name__ == "__main__":
    model_path = './eye_openness_classifier_1_BEAT.tflite'
    model_interpreter = build_model(model_path)

    # input_details = model_interpreter.get_input_details()
    # output_details = model_interpreter.get_output_details()

    '''
    ver1
    input_details [{'name': 'input', 'index': 0, 'shape': array([ 1, 36, 60,  3], dtype=int32), 'shape_signature': array([-1, 36, 60,  3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    output_details [{'name': 'Identity', 'index': 152, 'shape': array([1, 1], dtype=int32), 'shape_signature': array([-1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'Identity', 'index': 152, 'shape': array([1, 1], dtype=int32), 'shape_signature': array([-1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    ver2
    input_details [{'name': 'serving_default_args_0:0', 'index': 0, 'shape': array([ 1, 36, 60,  3], dtype=int32), 'shape_signature': array([ 1, 36, 60,  3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    output_details [{'name': 'StatefulPartitionedCall:0', 'index': 157, 'shape': array([1, 3], dtype=int32), 'shape_signature': array([1, 3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
    '''

    R_dec_COUNTER = 0
    R_dec_TOTAL = 0

    L_dec_COUNTER = 0
    L_dec_TOTAL = 0
    
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_threshold =  int(0.1/(1/fps))

    time_per_frame_video = 1/fps
    last_time = time.perf_counter()

    while(True):
        ret, img = cap.read()

        # fsp 계산
        # time_per_frame = time.perf_counter() - last_time
        # time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
        # time.sleep(time_sleep_frame)

        real_fps = 1/(time.perf_counter()-last_time)
        last_time = time.perf_counter()
        
        frame_threshold =  int(real_fps * 0.1)
        if frame_threshold == 0:
            frame_threshold = 1
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        try :
            img_ori = img.copy()
            img_show, left_landmark, right_landmark = detect_face_landmark(img_ori)
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            
            if VER == 'ver1':
                R_pred, L_pred, r_img, l_img = model_run(model_interpreter, left_landmark, right_landmark, img_ori)
                
                r_img = cv2.resize(r_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                l_img = cv2.resize(l_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

                eye_img = cv2.hconcat([r_img, l_img])

                if R_pred < 0.5:
                    R_dec_COUNTER += 1
                else:
                    if R_dec_COUNTER >= frame_threshold :
                        R_dec_TOTAL += 1
                    # reset the eye frame counter
                    R_dec_COUNTER = 0

                if L_pred < 0.5:
                    L_dec_COUNTER += 1
                else:
                    if L_dec_COUNTER >= frame_threshold:
                        L_dec_TOTAL += 1
                    # reset the eye frame counter
                    L_dec_COUNTER = 0

                cv2.putText(img_show, "real_fps : " + str(round(real_fps, 2)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(img_show, "R_blink : " + str(R_pred), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 if R_pred > 0.5 else 255, 0, 255 if R_pred > 0.5 else 0), 2, cv2.LINE_AA)
                cv2.putText(img_show, "R_blink_num : " + str(R_dec_TOTAL), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(img_show, "L_blink : " + str(L_pred), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 if L_pred > 0.5 else 255, 0, 255 if L_pred > 0.5 else 0), 2, cv2.LINE_AA)
                cv2.putText(img_show, "L_blink_num : " + str(L_dec_TOTAL), (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                img_show[-eye_img.shape[0]:, :eye_img.shape[1]] = eye_img
                cv2.imshow('detect_face_mesh', img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if VER == 'ver2':
                R_pred1, R_pred2, R_pred3, L_pred1, L_pred2, L_pred3, r_img, l_img = model_run(model_interpreter, left_landmark, right_landmark, img_ori)
                
                R_pred_ls =[R_pred1, R_pred2, R_pred3]
                r_pred_result = max(R_pred_ls)
                R_pred_idx_result = R_pred_ls.index(r_pred_result)

                L_pred_ls =[L_pred1, L_pred2, L_pred3]
                L_pred_result = max(L_pred_ls)
                L_pred_idx_result = L_pred_ls.index(L_pred_result)

                r_img = cv2.resize(r_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                l_img = cv2.resize(l_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

                eye_img = cv2.hconcat([r_img, l_img])

                cv2.putText(img_show, "real_fps : " + str(round(real_fps, 2)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if R_pred_idx_result == 0:
                    cv2.putText(img_show, "R_open " + str(r_pred_result), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if R_pred_idx_result == 1:
                    cv2.putText(img_show, "R_close " + str(r_pred_result), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if R_pred_idx_result == 2:
                    cv2.putText(img_show, "R_Not_eye " + str(r_pred_result), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if L_pred_idx_result == 0:
                    cv2.putText(img_show, "L_open " + str(L_pred_result), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if L_pred_idx_result == 1:
                    cv2.putText(img_show, "L_close " + str(L_pred_result), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if L_pred_idx_result == 2:
                    cv2.putText(img_show, "L_Not_eye " + str(L_pred_result), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                img_show[-eye_img.shape[0]:, :eye_img.shape[1]] = eye_img
                cv2.imshow('detect_face_mesh', img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except :
            cv2.imshow('detect_face_mesh', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
    cap.release()
    cv2.destroyAllWindows()