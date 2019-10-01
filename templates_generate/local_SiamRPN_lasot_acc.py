import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import os
import pdb

from net1 import SiamRPNBIG
from run_SiamRPN1 import SiamRPN_init, SiamRPN_track
from utils1 import get_axis_aligned_bbox,get_axis_aligned_rect, cxy_wh_2_rect, overlap_ratio

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()
reset = 1; frame_max = 300
setfile = 'update_set1'
temp_path = setfile+'_templates_step1_std'
if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

video_path = '/media/lichao/4data/tracking/datasets/LaSOTBenchmark'
lists = open('/home/lichao/tracking/LaSOT_Evaluation_Toolkit/sequence_evaluation_config/'+setfile+'.txt','r')
list_file = [line.strip() for line in lists]
category = os.listdir(video_path)
category.sort()

template_acc = []; template_cur = []
init0 = []; init = []; pre = []; gt = []  #init0 is reset init
for tmp_cat in category:
    videos = os.listdir(join(video_path, tmp_cat)); videos.sort()    
    for video in videos:
        if video not in list_file:
            continue
        print(video)        
        gt_path = join(video_path,tmp_cat,video, 'groundtruth.txt')
        ground_truth = np.loadtxt(gt_path, delimiter=',')
        num_frames = len(ground_truth);  #num_frames = min(num_frames, frame_max)
        img_path = join(video_path,tmp_cat,video, 'img');
        imgFiles = [join(img_path,'%08d.jpg') % i for i in range(1,num_frames+1)]
        frame = 0;
        while frame < num_frames:
            Polygon = ground_truth[frame]
            cx, cy, w, h = get_axis_aligned_bbox(Polygon)
            if w*h!=0:
                image_file = imgFiles[frame]
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                im = cv2.imread(image_file)  # HxWxC
                state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
                template_acc.append(state['z_f']); template_cur.append(state['z_f_cur'])
                init0.append(0); init.append(frame); frame_reset=0;pre.append(0); gt.append(1)
                while frame < (num_frames-1):
                    frame = frame + 1; frame_reset=frame_reset+1
                    image_file = imgFiles[frame]
                    if not image_file:
                        break
                    im = cv2.imread(image_file)  # HxWxC
                    state = SiamRPN_track(state, im)  # track
                    #pdb.set_trace()
                    template_acc.append(state['z_f']); template_cur.append(state['z_f_cur'])
                    init0.append(frame_reset); init.append(frame); pre.append(1); 
                    if frame==(num_frames-1): #last frame
                        gt.append(0)
                    else:
                        gt.append(1)
                    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    if reset:                    
                        gt_rect = get_axis_aligned_rect(ground_truth[frame])
                        iou = overlap_ratio(gt_rect, res)
                        if iou<=0:
                            break    
            else:
                template_acc.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32));  
                template_cur.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32));        
                init0.append(0); init.append(frame); pre.append(1); 
                if frame==(num_frames-1): #last frame
                    gt.append(0)
                else:
                    gt.append(1)           
            frame = frame + 1 #skip
template_acc=np.concatenate(template_acc); template_cur=np.concatenate(template_cur)
np.save(temp_path+'/template',template_acc); np.save(temp_path+'/templatei',template_cur)
np.save(temp_path+'/init0',init0); np.save(temp_path+'/init',init);np.save(temp_path+'/pre',pre);np.save(temp_path+'/gt',gt);
