import argparse

import os

from torchvision.models.resnet import List
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models

from numpy import random
from google.colab.patches import cv2_imshow
from IPython.display import display

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

import cv2
import numpy as np

from easydict import EasyDict
from random import randint
from imutils.video import FPS

from keras.models import load_model

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.55,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=[0],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    args = EasyDict({

        'detector': "yolov3",

        # Path Params
        'videoPath': "/content/Basketball-Action-Recognition/dataset/examples/0031131.mp4",

        # Player Tracking
        'classes': ["person"],
        'tracker': "CSRT",
        'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
        'singleTracker': True,

        # Court Line Detection
        'draw_line': False,

        # YOLOV3 Detector
        'weights': "yolov3.weights",
        'config': "yolov3.cfg",

        'COLORS': np.random.uniform(0, 255, size=(1, 3)),

        # Action Recognition
        'base_model_name': 'r2plus1d_multiclass',
        'pretrained': True,
        'lr': 0.0001,
        'start_epoch': 2,
        'num_classes': 3,
        'labels': {"0" : "pick", "1" : "stand", "2" : "walk"},
        'model_path': "model_checkpoints/r2plus1d_augmented-2/",
        'history_path': "histories/history_r2plus1d_augmented-2.txt",
        'seq_length': 16,
        'vid_stride': 8,
        'output_path': "output_videos/"

    })

    result_list = []
    frames_person = {}
    frames_person_id = {}
    all_frames = read_video_frames(str(source))
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size
    
    # Load model Action Recognition
    list_model = {}

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset.sources)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, [0], agnostic_nms)
        dt[2] += time_synchronized() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]

                        #result_list.append([frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i])
                        
                        # new code
                        #print(frame_idx,bbox_left, bbox_top, bbox_w, bbox_h)
                        #print(type(frame_idx), type(bbox_left), type(bbox_top), type(bbox_w), type(bbox_h))
                        #print('all_frames len:', len(all_frames))
                        croped_frame = crop_frame(all_frames[frame_idx], int(bbox_left), int(bbox_top), int(bbox_w), int(bbox_h))

                        resized_frame = resize_image_maintain_aspect_ratio(croped_frame)
                        #resized_frames.append(resized_frame)
                        #cv2.imwrite(f"/content/img/frame_{id}_{fi}.jpg", resized_frame)

                        if id not in frames_person or frames_person[id] is None:
                          #print ('None')
                          new_arr = []
                          new_arr.append(resized_frame)
                          frames_person[id] = new_arr

                          frames_id = []
                          frames_id.append(frame_idx)
                          frames_person_id[id] = frames_id
                        else:
                          #print('OK')  
                          existing_arr = frames_person[id]
                          existing_arr.append(resized_frame)
                          frames_person[id] = existing_arr

                          existing_arr_id = frames_person_id[id]
                          existing_arr_id.append(frame_idx)
                          frames_person_id[id] = existing_arr_id
                        
                        # Write MOT compliant results to file
                        #with open(txt_path + '.txt', 'a') as f:
                        #    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                        #                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            action = '-'

                            # Prediction action
                            predictions = {}
                            if id in frames_person and len(frames_person[id]) > 15:
                              frames_of_person = frames_person[id]
                              frames = select_items_with_equal_spacing(frames_of_person, 16, 3)

                              frames_of_person_id = frames_person_id[id]
                              frames_selected = select_items_with_equal_spacing(frames_of_person_id, 16, 3)
                              
                              if len(frames) > 15:
                                print('\n['+str(id)+'] frames_selected:',frames_selected)
        
                                frames = np.asarray(frames)
                                #print(frames.shape)
                                #frames = np.expand_dims(frames, axis=0)
                                #print(frames.shape)

                                #input_frames = inference_batch(torch.FloatTensor(frames))
                                #print('input_frames len:', len(frames))
                                #print(input_frames.shape)

                                #input_frames = input_frames.to(device=device)

                                action_model = None
                                with torch.no_grad():
                                    if id in list_model:
                                      action_model = list_model[id]
                                    else:
                                      action_model = Load_LRCN_model()
                                      list_model[id] = action_model

                                    # Predict
                                    predicted_labels_probabilities = action_model.predict(np.expand_dims(frames, axis=0))[0]
                                    predicted_label = np.argmax(predicted_labels_probabilities)

                                    CLASSES_LIST = ["pick", "stand", "walk"]
                                    action = CLASSES_LIST[predicted_label]

                                #print(predsx.cpu().numpy().tolist())
                                #predictions[id] = predsx.cpu().numpy().tolist()
                                #print('predict id = ',id, 'action:', args.labels[str(predictions[id][0])],'\n')
                                print('predict id = ',id, 'action:', action,'\n')
                                #action = args.labels[str(predictions[id][0])]
                            
                            label = f'{id} {names[c]} : {action}'

                            #label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            #    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                #save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    return result_list

def read_video_frames(video_path):
    # เปิดวิดีโอ
    cap = cv2.VideoCapture(video_path)

    # ตรวจสอบว่าวิดีโอถูกเปิดหรือไม่
    if not cap.isOpened():
        print("Error: ไม่สามารถเปิดวิดีโอได้", video_path)
        return None

    # เก็บเฟรมของวิดีโอ
    frames = []

    # อ่านเฟรมทีละเฟรม
    while True:
        ret, frame = cap.read()

        # หากไม่มีเฟรมที่อ่านได้แล้ว (จบวิดีโอ)
        if not ret:
            break

        # เพิ่มเฟรมลงในรายการ
        frames.append(frame)

    # ปิดวิดีโอ
    cap.release()

    return frames

def crop_frame(frame, bbox_left, bbox_top, bbox_w, bbox_h):
    cropped_frame = frame[bbox_top:bbox_top+bbox_h, bbox_left:bbox_left+bbox_w]
    return cropped_frame

def resize_image_maintain_aspect_ratio(image, target_size=224):
    # ดึงขนาดปัจจุบันของรูปภาพ
    height, width, _ = image.shape

    # คำนวณอัตราส่วนของภาพที่คล้ายกัน
    aspect_ratio = width / height

    # ปรับขนาดรูปภาพให้มีความยาว (width) หรือความสูง (height) ตาม target_size
    if aspect_ratio > 1:  # ถ้าภาพนั้นกว้างกว่าที่สูง
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:  # ถ้าภาพนั้นสูงกว่าที่กว้าง
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # ปรับขนาดรูปภาพ
    resized_image = cv2.resize(image, (new_width, new_height))

    # สร้างรูปภาพที่มีขนาดตาม target_size และเติมสีดำ (padding)
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    return padded_image

def Load_LRCN_model():
    LRCN_model = load_model('/content/models/LRCN_model_Date_Time_2024_06_01_12_14_02_Loss0.28390315855158854_Accuracy_0.9534883720930233_n1.h5')
    return LRCN_model

def select_items_with_equal_spacing(lst, num_items):
    step = len(lst) // num_items
    return lst[::step][:num_items]

def select_items_with_equal_spacing(lst, num_items, step):
    start_index = max(len(lst) - num_items * step, 0)  # หาดัชนีเริ่มต้น
    return lst[start_index::step]  # ใช้การ slice และกำหนด step

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7-e6e.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt

def inference_batch(batch):
    # (batch, t, h, w, c) --> (batch, c, t, h, w)
    batch = batch.permute(0, 4, 1, 2, 3)
    return batch

def detect(vid_path):
  opt = parse_opt()
  opt.source = vid_path
  return run(**vars(opt))

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
