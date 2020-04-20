import socket
from multiprocessing import Process, Value, Queue
import heapq
import threading
import struct
import numpy as np
import cv2
import argparse
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

nextIndex = 0
data = []
heapq.heapify(data)
sharedState = Value('i', 1)
terminate = Queue()

parser = argparse.ArgumentParser(
description='''Lightweight human pose estimation python demo.
                This is just for quick results preview.
                Please, consider c++ demo for the best performance.''')
parser.add_argument('--checkpoint-path', type=str, default="checkpoint_iter_370000.pth", help='path to the checkpoint')
parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
parser.add_argument('--track', type=int, default=1, help='track pose id in video')
parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
args = parser.parse_args()

net = PoseEstimationWithMobileNet()
checkpoint = torch.load(args.checkpoint_path, map_location = 'cpu')
load_state(net, checkpoint)
net = net.eval()
if not args.cpu:
    net = net.cuda()

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not args.cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def socketComm(conn, index):
    global terminate
    global net
    width = struct.unpack("<I", conn.recv(4))[0]
    height = struct.unpack("<I", conn.recv(4))[0]
    conn.settimeout(60)
    print("Recieved webcam:", width, height)
    if (width>=3200) or (height>=2400):
        print("Too large, closing")
        conn.close()
        terminate.put(index)
        return


    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    delay = 33

    if True:
    #try:
        while True:
            datalen = conn.recv(4) #receives length of data before data
            if len(datalen) == 0:
                break
            datalen = struct.unpack("<I", datalen)[0] #convert little Endian binary to integer
            imageData = bytearray(0) #master bytearray (imageData) to append to
            while len(imageData) != datalen:
                imageData.extend(conn.recv(datalen - len(imageData)))

            imageData = np.frombuffer(imageData, dtype = np.uint8)
            #decodes binary from jpeg into an image we can work with
            img = cv2.imdecode(imageData, cv2.IMREAD_COLOR)
        
            orig_img = img.copy()
            heatmaps, pafs, scale, pad = infer_fast(net, img, args.height_size, stride, upsample_ratio, args.cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)
            
            response = struct.pack("<36i", *pose_keypoints.flatten())
            conn.send(struct.pack("<I",len(response)))
            conn.send(response)
    #except Exception as e:
    #    print(e)
    conn.close()
    print("Closing", index)
    terminate.put(index)

def watchDog(processes):
    global nextIndex
    global data
    global sharedState
    global terminate
    
    while sharedState.value:
        rem = terminate.get()
        if rem is None:
            return
        processes[rem].join()
        processes.pop(rem)
        heapq.heappush(data,rem)

def main():
    global nextIndex
    global data
    global sharedState
    global terminate
    
    processes = dict()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 50))
    #sock.bind(("127.0.0.1",9000))
    sock.listen()
    watchdog = threading.Thread(target=watchDog, args=(processes,))
    watchdog.start()
    print("Starting Server")
    try:
        while True:
            (conn,(ip,port)) = sock.accept()
            index = None
            if len(data) == 0:
                index = nextIndex
                nextIndex += 1
            else:
                index = heapq.heappop(data)
            processes[index] = Process(target=socketComm, args=(conn, index))
            processes[index].start()
    except:
        pass
    print("Shutting down")
    sharedState.value = 0
    sock.close()
    while len(processes) != 0:
        pass
    terminate.put(None)
    watchdog.join()

if __name__ == "__main__":
    main()
