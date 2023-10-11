"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import dota_utils as util
import re
import time
import pdb
import tqdm
import torch
import torchgeometry as tgm
from detectron2.layers import nms_rotated
## the thresh for nms when merge image
nms_thresh = 0.3

# singlenet with round sample
# threshold = {'roundabout': 0.3, 'tennis-court': 0.2, 'swimming-pool': 0.05, 'storage-tank': 0.3,
#                'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.2, 'plane': 0.3,
#                'large-vehicle': 0.3, 'helicopter': 0.1, 'harbor': 0.4, 'ground-track-field': 0.4,
#                'bridge': 0.2, 'basketball-court': 0.1, 'baseball-diamond': 0.5}

# singlenet with ellipse sample
threshold = {'roundabout': 0.3, 'tennis-court': 0.2, 'swimming-pool': 0.05, 'storage-tank': 0.3,
               'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.2, 'plane': 0.3,
               'large-vehicle': 0.2, 'helicopter': 0.1, 'harbor': 0.1, 'ground-track-field': 0.3,
               'bridge': 0.1, 'basketball-court': 0.1, 'baseball-diamond': 0.5,
            'container-crane': nms_thresh, 'airport': nms_thresh, 'helipad': nms_thresh}

# threshold = {'roundabout': nms_thresh, 'tennis-court': nms_thresh, 'swimming-pool': nms_thresh, 'storage-tank': nms_thresh,
#             'soccer-ball-field': nms_thresh, 'small-vehicle': nms_thresh, 'ship': nms_thresh, 'plane': nms_thresh,
#             'large-vehicle': nms_thresh, 'helicopter': nms_thresh, 'harbor': nms_thresh, 'ground-track-field': nms_thresh,
#             'bridge': nms_thresh, 'basketball-court': nms_thresh, 'baseball-diamond': nms_thresh}

# threshold = {'roundabout': nms_thresh, 'tennis-court': nms_thresh, 'swimming-pool': nms_thresh, 'storage-tank': nms_thresh,
#                 'soccer-ball-field': nms_thresh, 'small-vehicle': nms_thresh, 'ship': nms_thresh, 'plane': nms_thresh,
#                 'large-vehicle': nms_thresh, 'helicopter': nms_thresh, 'harbor': nms_thresh, 'ground-track-field': nms_thresh,
#                 'bridge': nms_thresh, 'basketball-court': nms_thresh, 'baseball-diamond': nms_thresh,
#                 'container-crane': nms_thresh, 'airport': nms_thresh, 'helipad': nms_thresh}

def batch_polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = torch.stack([bbox[:, 0::2], bbox[:, 1::2]], dim=1)
    angle = torch.atan2(-(bbox[:, 0,1]-bbox[:, 0,0]), 
                        bbox[:, 1,1]-bbox[:, 1,0])

    center = torch.zeros(bbox.size(0), 2, 1, dtype=bbox.dtype, device=bbox.device)

    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = center / 4.0
    R = torch.stack([torch.cos(angle), -torch.sin(angle), 
                     torch.sin(angle), torch.cos(angle)], dim=1)
    R = R.reshape(-1, 2, 2)

    normalized = torch.matmul(R.transpose(2,1), bbox - center)
    
    if bbox.size(0) == 0:
        return torch.empty((0, 5), dtype=bbox.dtype, device=bbox.device)

    xmin = torch.min(normalized[:, 0,:], dim=1)[0]
    xmax = torch.max(normalized[:, 0,:], dim=1)[0]
    ymin = torch.min(normalized[:, 1,:], dim=1)[0]
    ymax = torch.max(normalized[:, 1,:], dim=1)[0]

    w = xmax - xmin 
    h = ymax - ymin

    center = center.squeeze(-1)
    center_x = center[:, 0]
    center_y = center[:, 1]
    new_box = torch.stack([center_x, center_y, w, h, -tgm.rad2deg(angle)], dim=1)
    return new_box


def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def detectron2_rota_nms(dets, cls_name):
    
    keep = None
    # try:
    scores = dets[:, 8]
    thresh = float(threshold[cls_name])
    rbox = batch_polygonToRotRectangle(torch.from_numpy(dets[:, 0:8])).to(torch.device('cuda'))
    keep = nms_rotated(rbox, torch.from_numpy(scores).to(torch.device('cuda')), thresh)
    keep = keep.to(torch.device('cpu')).numpy()

    
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, cat_name):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in tqdm.tqdm(nameboxdict):
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), cat_name)
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergebase(srcpath, dstpath, nms, dota=1):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        name = util.custombasename(fullname)
        # print('name:', name)
        dstname = os.path.join(dstpath, name + '.txt')
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            lines = f_in.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            print("Runninng file merge: {}, numbers: {}".format(name, len(splitlines)))
            for splitline in splitlines:
                subname = splitline[0]
                splitname = subname.split('__')
                oriname = splitname[0]
                pattern1 = re.compile(r'__\d+___\d+')
                #print('subname:', subname)
                x_y = re.findall(pattern1, subname)
                x_y_2 = re.findall(r'\d+', x_y[0])
                x, y = int(x_y_2[0]), int(x_y_2[1])

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

                rate = re.findall(pattern2, subname)[0]

                confidence = splitline[1]
                poly = list(map(float, splitline[2:]))
                origpoly = poly2origpoly(poly, x, y, rate)
                det = origpoly
                det.append(confidence)
                det = list(map(float, det))
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []
                nameboxdict[oriname].append(det)
                cat_name = name.split('_')[-1]
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, cat_name)
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:
                        #print('det:', det)
                        confidence = det[-1]
                        bbox = det[0:-1]
                        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                        if dota == 1:
                            line = f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f}"
                        if dota == 2:
                            line = f"{x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f} {x3:.0f} {y3:.0f} {x4:.0f} {y4:.0f}"
                        outline = imgname + ' ' + str(confidence) + ' ' + line
                        #print('outline:', outline)
                        f_out.write(outline + '\n')
def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath, dota=1):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    mergebase(srcpath,
              dstpath,
              detectron2_rota_nms, dota)
if __name__ == '__main__':
    # see demo for example
    mergebypoly(r'path_to_configure', r'path_to_configure')
    # mergebyrec()