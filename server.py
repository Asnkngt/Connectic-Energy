import socket
from multiprocessing import Process, Value, Queue
import heapq
import threading
import struct
import numpy as np
import cv2
import tensorflow as tf
import posenet
import argparse

nextIndex = 0
data = []
heapq.heapify(data)
sharedState = Value('i', 1)
terminate = Queue()

def socketComm(conn, index):
    global terminate
    width = struct.unpack("<I", conn.recv(4))[0]
    height = struct.unpack("<I", conn.recv(4))[0]
    conn.settimeout(60)
    print("Recieved webcam:", width, height)
    if (width>=3200) or (height>=2400):
        print("Too large, closing")
        conn.close()
        terminate.put(index)
        return

    target_width, target_height = posenet.valid_resolution(width, height)
    scale = np.array([height / target_height, width / target_width])

    #windowName = "image " + str(index)
    #cv2.namedWindow(windowName) #creates a window named "image"

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=int, default=101)
        parser.add_argument('--cam_width', type=int, default=width)
        parser.add_argument('--cam_height', type=int, default=height)
        parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
        args = parser.parse_args()
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(args.model, sess)
            output_stride = model_cfg['output_stride']
            
            while sharedState.value:
                datalen = conn.recv(4) #receives length of data before data
                if len(datalen) == 0:
                    break
                datalen = struct.unpack("<I", datalen)[0] #convert little Endian binary to integer
                imageData = bytearray(0) #master bytearray (imageData) to append to
                while len(imageData) != datalen:
                    imageData.extend(conn.recv(datalen - len(imageData)))

                imageData = np.frombuffer(imageData, dtype = np.uint8)
                #decodes binary from jpeg into an image we can work with
                image = cv2.imdecode(imageData, cv2.IMREAD_COLOR)
            
                #display "video"
                #cv2.imshow(windowName, image) #finds window specificed (parameter 1) and displays img on image
                #cv2.waitKey(1)
            
                input_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                input_img = input_img * (2.0 / 255.0) - 1.0
                input_img = input_img.reshape(1, target_height, target_width, 3)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_img}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=1,
                    min_pose_score=0.25)

                #print(pose_scores, keypoint_scores, keypoint_coords)
                #overlay_image = posenet.draw_skel_and_kp(
                #image, pose_scores, keypoint_scores, keypoint_coords,
                #min_pose_score=0.15, min_part_score=0.1)

                response = struct.pack("<51f", *keypoint_scores.flatten(), *keypoint_coords.flatten())

                conn.send(struct.pack("<I",len(response)))
                conn.send(response)
    except Exception as e:
        print(e)
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
    sock.bind(("0.0.0.0",9000))
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
