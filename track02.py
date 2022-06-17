# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import copy
import torch.backends.cudnn as cudnn
import numpy as np

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, myheatmap, id_color, is_in_poly
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


source = r'E:\darklabel\data\videoandlabel\2022_03_01_00_00_00/2022_03_01_00_00_00.mp4'
cap = cv2.VideoCapture(source)
ret, img = cap.read()
eatpolypoints = []
drinkpolypoints = []
eatpolys = []
drinkpolys = []

# Mouse event left button choose eat area points, right button choose drink area points,
# middle button save points.
def draw_roi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        point = (x, y)
        eatpolypoints.append(point)
        print("Eat Poly Points" + str(eatpolypoints))
        cv2.circle(img, (x, y), 4, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=2)
        cv2.imshow("first_frame", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        point = (x, y)
        drinkpolypoints.append(point)
        print("Drink Poly Points" + str(drinkpolypoints))
        cv2.circle(img, (x, y), 4, (0, 255, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=2)
        cv2.imshow("first_frame", img)
    elif event == cv2.EVENT_MBUTTONDOWN:
        if len(eatpolypoints) >= 3:
            eatpolys.append(copy.deepcopy(eatpolypoints))
        if len(drinkpolypoints) >= 3:
            drinkpolys.append(copy.deepcopy(drinkpolypoints))
        eatpolypoints.clear()
        drinkpolypoints.clear()
        print('Eat areas: ' + str(eatpolys))
        print('Drink areas: ' + str(drinkpolys))


cv2.namedWindow("first_frame")
cv2.setMouseCallback("first_frame", draw_roi)
cv2.imshow("first_frame", img)
cv2.waitKey(0)


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_heat, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_heat, opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    id_centers = [deque(maxlen=30) for _ in range(5000)]
    centers = []

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string
            # Call annotator
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                in_eating_area = 0
                in_drinking_area = 0
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        color = id_color(id)
                        center = (int(((bboxes[0]) + (bboxes[2])) / 2), int(((bboxes[1]) + (bboxes[3])) / 2))
                        id_centers[id].append(center)
                        centers.append(center)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        # Draw area for eating and drinking
                        for epoly in eatpolys:
                            # print(len(epoly))
                            for point_index in range(len(epoly)):
                                if point_index + 1 < len(epoly) and len(epoly)>=3:
                                    cv2.line(im0, epoly[point_index], epoly[point_index + 1],
                                             (0, 0, 255), 2)
                                else:
                                    cv2.line(im0, epoly[point_index], epoly[0], (0, 0, 255), 2)
                            cv2.putText(im0, 'Eating area', epoly[1], cv2.FONT_HERSHEY_PLAIN,
                                        2.0, (0, 0, 0), thickness=2)
                        for dpoly in drinkpolys:
                            for point_index in range(len(dpoly)):
                                if point_index + 1 < len(dpoly) and len(dpoly)>=3:
                                    cv2.line(im0, dpoly[point_index], dpoly[point_index + 1],
                                             (0, 255, 0), 2)
                                else:
                                    cv2.line(im0, dpoly[point_index], dpoly[0], (0, 255, 0), 2)
                            cv2.putText(im0, 'Drinking area', dpoly[1], cv2.FONT_HERSHEY_PLAIN,
                                        2.0, (0, 0, 0), thickness=2)
                        for poly in eatpolys:
                            if is_in_poly(center, poly):
                                in_eating_area = in_eating_area + 1
                                annotator.box_label(bboxes, label, color)
                        for poly in drinkpolys:
                            if is_in_poly(center, poly):
                                in_drinking_area = in_drinking_area + 1
                                annotator.box_label(bboxes, label, color)
                        for j in range(1, len(id_centers[id])):
                            if id_centers[id][j - 1] is None or id_centers[id][j] is None:
                                continue
                            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                            cv2.line(im0, (id_centers[id][j - 1]), (id_centers[id][j]), color, thickness)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                    # Add detection information to the video
                    cv2.putText(im0, 'Broiler number: ' + str(len(outputs)),
                                (20, 40), 0, 1.2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(im0, 'Density: ' + str(round(len(outputs), 2)) + 'area' + 'per square meter',
                                (20, 80), 0, 1.2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(im0, 'FPS: ' + str(round(1 / ((t3 - t2) + (t5 - t4)), 2)),
                                (20, 120), 0, 1.2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(im0, 'Broilers in eating area: ' + str(in_eating_area),
                                (20, 160), 0, 1.2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(im0, 'Broilers in drinking area: ' + str(in_drinking_area),
                                (20, 200), 0, 1.2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            heat_img = myheatmap(im0, centers, 0.4)

            if show_vid:
                cv2.namedWindow('heat')
                cv2.imshow(str(p), im0)
                cv2.imshow('heat', heat_img)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # if save_heat:

            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default=r'C:\Users\user\Desktop\projects\yolov5-5.0\runs\train\exp7\weights\best.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-heat', action='store_true', help='save heat video results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is broiler
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)

