import time

# Reid Part
from utils import *
from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg

# YOLO Part
from models.experimental import attempt_load
from utils_yolo.datasets import LoadStreams, LoadImages
from utils_yolo.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils_yolo.torch_utils import select_device, load_classifier, time_synchronized


def detect(view_img,
           save_txt,
           imgsz,
           device1,
           augment1,
           agnostic_nms1,
           classes1,
           iou_thres1,
           conf_thres1,
           output='inference/output',
           source='inference/images',
           weights='weights/yolov5/yolov5s.pt',
           ):
    # output, source, weights, view_img, save_txt, imgsz = \
    #     opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    save_img = False
    dist_threshold = 1.0

    # ReID Part
    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Reid Model
    query_loader, num_query = make_data_loader(reidCfg)
    reid_model = build_model(reidCfg, num_classes=10126)
    reid_model.load_param(reidCfg.TEST.WEIGHT)  # weights/reID/719rank1.pth  pre-trained weights
    reid_model.to(device).eval()
    print('successful')

    query_feats = []   # features 的query
    query_pids  = []

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch
            img = img.to(device)
            feat = reid_model(img)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats.append(feat)
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2) # 计算出查询图片的特征向量
    print('the query feats:', query_feats)

    # YOLO Part
    # Initialize
    set_logging()
    device = select_device(device1)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    number_person = 0

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, im2, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment1)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres1, iou_thres1, classes=classes1, agnostic=agnostic_nms1)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(output) / Path(p).name)
            txt_path = str(Path(output) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                gallery_img = []
                gallery_loc = []

                for *xyxy, conf, cls in reversed(det):
                    if cls == 0:  # When the class is person
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            x1 = int(xyxy[0])
                            y1 = int(xyxy[1])
                            x2 = int(xyxy[2])
                            y2 = int(xyxy[3])
                            w = x2 - x1
                            h = y2 - y1

                            # Obtain the person image
                            if w*h > 500:
                                gallery_loc.append((x1, y1, x2, y2))
                                crop_img = im2[y1:y2, x1:x2]
                                crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                                crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)
                                gallery_img.append(crop_img)
                                # save path and name
                                # save_path1 = r'E:\.Cstation\PycharmPlace\Final_Year_Project\inference\buffer\\' + str(
                                #     number_person) + '.jpg'
                                # cv2.imwrite(save_path1, im1)
                                # number_person += 1
                                # im1

                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reid_model(gallery_img)
                    print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                    # m: 2
                    # n: 7
                    # 这里是在做相似度度量吧
                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                    distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                    distmat = distmat.sum(axis=0) / len(query_feats)

                    index = distmat.argmin()

                    if distmat[index] < dist_threshold:
                        print('Distance：%s' % distmat[index])
                        plot_one_box(gallery_loc[index], im0, label='find!', color=(0, 0, 255))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(output))

    print('Done. (%.3fs)' % (time.time() - t0))