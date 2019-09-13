'''
Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os
import random

#
# pathIn = "./mica_hand_videos/"
# pathOut = "./data/actions/"
# actions = ["action4", "action5", "action8", "action9"]
#
# try:
#     if not os.path.exists(pathOut):
#         os.makedirs(pathOut)
# except OSError:
#     print ('Error: Creating directory of data')
#
# for action_name in actions :
#     scan_directory = os.path.join(pathIn, action_name)
#     for index, file_name in enumerate(sorted([name for name in os.listdir(scan_directory)
#             if os.path.isfile(os.path.join(scan_directory, name))])):
#
#         video_path = os.path.join(scan_directory, file_name)
#         print(video_path)
#         video_cap = cv2.VideoCapture(video_path)
#         frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
#         milisecond_per_frame = int((1 / frames_per_second) * 1000)
#         print("milisecond_per_frame = ", milisecond_per_frame)
#
#         num_frames = (int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)  # actually having less than 1 frame
#         # xet them truong hop neu num_frames < 16
#         chosen_frames = list(np.linspace(1, num_frames, 16, dtype=np.int))
#         list_images = []
#
#         for frame_id in chosen_frames:
#             video_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id * milisecond_per_frame))
#             success_state, image = video_cap.read()
#             if (success_state == True):
#                 resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
#                 list_images.append(resized)
#
#         print("num_frames = ", num_frames)
#         print("chosen_frames = ", chosen_frames)
#         print("==> Save long image")
#
#         long_image = np.concatenate(list_images, axis=1)
#         print("long_image.shape = ", long_image.shape)
#         cv2.imwrite(pathOut + "%s/%04d.png" % (action_name, index+1), np.array(long_image))
#



# performers = ["001_Giang", "002_VuHai", "003_NguyenTrongTuyen", "004_TranDucLong", "005_TranThiThuThuy",
#               "006_KhongVanMinh", "007_BuiHaiPhong", "008_NguyenThiThanhNhan", "009_Binh", "010_Tan",
#               "011_Thuan"]

# views = ["0001", "0002", "0003", "0004", "0005"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009",
#           "0010", "0011", "0012"]
# root_path = "/home/dungbachviet/Desktop/mica_hand_videos"
# destinatation_path = "./action_view_data/"










######## 6 PERFORMERS #######################



# ###########################################3
# performers = ["001_Giang"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
# destination_path = "./action_view_data_giang_6dau/"
#
# performers = ["001_Giang"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_giang_6cuoi/"
#
# ###########################################



###########################################3
# performers = ["002_VuHai"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
# destination_path = "./action_view_data_hai_6dau/"

# performers = ["002_VuHai"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_hai_6cuoi/"
#
# ###########################################


# ###########################################3
performers = ["004_TranDucLong"]
actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
destination_path = "./action_view_data_long_6dau/"
#
# performers = ["004_TranDucLong"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_long_6cuoi/"
#
# ###########################################


# ###########################################3
# performers = ["006_KhongVanMinh"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
# destination_path = "./action_view_data_minh_6dau/"
#
# performers = ["006_KhongVanMinh"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_minh_6cuoi/"
#
# ###########################################


# ###########################################3
# performers = ["005_TranThiThuThuy"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
# destination_path = "./action_view_data_thuy_6dau/"
#
# performers = ["005_TranThiThuThuy"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_thuy_6cuoi/"
#
# ###########################################


# ###########################################3
# performers = ["003_NguyenTrongTuyen"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006"]
# destination_path = "./action_view_data_tuyen_6dau/"
#
# performers = ["003_NguyenTrongTuyen"]
# actions = ["0007", "0008", "0009", "0010", "0011", "0012"]
# destination_path = "./action_view_data_tuyen_6cuoi/"
#
# ###########################################







views = ["0001", "0002", "0003", "0004", "0005"]
root_path = "/home/dungbachviet/Desktop/mica_hand_videos"


# Create directories of format : viewId_actionId
for view_id in views:
    for action_id in actions:
        created_path = os.path.join(destination_path, "%s_%s" % (view_id, action_id))
        if (not os.path.exists(created_path)):
            os.makedirs(created_path)

# Reconstruct data and move to destination path
for view_id in views:
    for action_id in actions:
        count_video = 0 # count number of videos in the same label (view + action)
        for performer_id in performers:

            # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang
            if (not os.path.exists(os.path.join(root_path, "%s" % performer_id))):
                print("Not exist path to %s", os.path.join(root_path, "%s" % performer_id))
                continue

            # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001
            if (not os.path.exists(os.path.join(root_path, "%s/%s" % (performer_id, view_id)))):
                print("Not exist path to %s", os.path.join(root_path, "%s/%s" % (performer_id, view_id)))
                continue


            # path to view :/home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001
            scan_directory = os.path.join(root_path, "%s/%s" % (performer_id, view_id))
            for folder_name in sorted([name for name in os.listdir(scan_directory) if os.path.isdir(os.path.join(scan_directory, name))]):
                print("\n\nfolder_name : ", folder_name)
                action = folder_name.strip().split("_") # action_performNum
                print("file_name after splitting : ", action)

                if (int(action[0]) == int(action_id)):
                    # Get out a video to convert to a "long image"
                    video_path = os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi"))

                    # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001/1_1/video.avi
                    if (not os.path.exists(os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi")))):
                        print("Not exist path to %s", os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi")))
                        continue

                    print("video_path = ", video_path)

                    # object to manage information of video
                    video_cap = cv2.VideoCapture(video_path)
                    frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
                    print("frames_per_second = ", frames_per_second)

                    milisecond_per_frame = int((1 / frames_per_second) * 1000)
                    print("milisecond_per_frame = ", milisecond_per_frame)

                    num_frames = (int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)  # actually having less than 1 frame
                    print("num_frames = ", num_frames)

                    # 3 strategies to split one video: 1 equal split, 2 random split
                    for strategy_index in range(3):
                        # Truong hop neu num_frames < 16 ==> automatically duplicate one frame multiple times
                        chosen_frames = list(np.linspace(1, num_frames, 16, dtype=np.int))
                        print("chosen_frames = ", chosen_frames)

                        if (strategy_index > 0):
                            # interval = int((num_frames - 16) / (15 * 2))
                            interval = int((num_frames - 16) / (15))
                            print("interval = ", interval)
                            if interval <= 0 : continue

                            # when interval > 0 ==> can random to split
                            for i in range(16):
                                if (i == 0 or i == 15): continue
                                chosen_frames[i] += random.randint(-interval, interval)

                            print("chosen_frames (update) = ", chosen_frames)


                        # Count number of split videos
                        count_video += 1

                        # Save frames split from videos
                        list_images = []
                        for frame_id in chosen_frames:
                            # Refer to the location to get the expected frame
                            # One frame can be got out multiple times (if total frames less than 16)
                            video_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id * milisecond_per_frame))
                            success_state, image = video_cap.read()

                            # Resize each frame to size of (64,64)
                            if (success_state == True):
                                resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
                                list_images.append(resized)

                        print("==> Save long image")
                        # Concatenate all frames into a "long image"
                        long_image = np.concatenate(list_images, axis=1)
                        print("long_image.shape = ", long_image.shape)

                        # path to save the "long image" of a video
                        save_video_to = os.path.join(destination_path, "%s_%s/%04d.png" % (view_id, action_id, count_video))
                        cv2.imwrite(save_video_to, np.array(long_image))

