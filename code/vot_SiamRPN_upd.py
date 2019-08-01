import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net_upd import SiamRPNBIG
from updatenet import UpdateResNet
from run_SiamRPN_upd import SiamRPN_init, SiamRPN_track_upd
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()
updatenet = UpdateResNet()    
update_model=torch.load('../models/vot2016.pth.tar')['state_dict']
#update_model_fix = dict()
#for i in update_model.keys():
#    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
#updatenet.load_state_dict(update_model_fix)
updatenet.load_state_dict(update_model)
updatenet.eval().cuda()
# warm up
#for i in range(10):
#    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
#    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_file)  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC

    state = SiamRPN_track_upd(state, im,updatenet)
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))

