import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import *

max_magnitude = 0
# def array_to_png(parms, array, filename="output.png"):
#     out_img = array/150
#     image = np.uint8(255-(out_img*255))
#     return image
#     cv2.imwrite(filename, image)

def viz(flo):
    global max_magnitude
    magnitude = np.linalg.norm(flo, axis=0)
    magnitude = np.expand_dims(magnitude, axis=0)
    if(np.max(magnitude) > max_magnitude):
        max_magnitude = np.max(magnitude)
        print(max_magnitude)
    # blur_map = magnitude[0]/150
    # blur_map = np.uint8(255-(blur_map*255))
    flo = flo.transpose((1,2,0))
    flo = flow_viz.flow_to_image(flo)
    return flo, magnitude[0]

def load_image(imfile, device='cuda'):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def get_blur_map(model, root_path, sharp_list):
    with torch.no_grad():
        flow_foward = 0
        flow_backward = 0
        for i in range(0, len(sharp_list)-1, 1):
            img1 = load_image(os.path.join(root_path, sharp_list[i]))
            img2 = load_image(os.path.join(root_path, sharp_list[i+1]))
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
    
            flow_low_1n, flow_up_1n = model(img1, img2, iters=20, test_mode=True)
            
            flow_foward += flow_up_1n
        for j in range(len(sharp_list)-1, 0, -1):
            img1 = load_image(os.path.join(root_path, sharp_list[j]))
            img2 = load_image(os.path.join(root_path, sharp_list[j-1]))
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
    
            flow_low_1n, flow_up_1n = model(img1, img2, iters=20, test_mode=True)
            
            flow_backward -= flow_up_1n
        flow_foward = flow_foward.cpu().squeeze().numpy()
        flow_backward = flow_backward.cpu().squeeze().numpy()
        flow = (flow_foward + flow_backward) / 2
        flow_img, blur_map = viz(flow)
        return flow_img, blur_map

def get_avg_blur(root_path, sharp_list):
    sum_image = None

    for path in sharp_list:
        path = os.path.join(root_path, path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            result = f"Failed to read image at: {path}"
            break

        if sum_image is None:
            sum_image = np.zeros_like(image, dtype=np.float32)
        
        if image.shape != sum_image.shape:
            result = "Image sizes are different. Cannot compute average."
            break

        # 累加圖片，注意轉換成 float32 以避免數據溢出
        sum_image += image.astype(np.float32)

    # 計算平均值並保存結果
    if sum_image is not None:
        average_image = (sum_image / len(sharp_list)).astype(np.uint8)
        return average_image
    return None

def building_out_video_folder(parms, video):
    # construct output video folder and the corresponding folder
    output_video_folder = os.path.join(parms['output_folder'],video)
    os.makedirs(output_video_folder, exist_ok=True)
    output_blur_img_folder = os.path.join(output_video_folder,"blur_img")
    os.makedirs(output_blur_img_folder, exist_ok=True)
    output_flow_img_folder = os.path.join(output_video_folder, "flow_img")
    os.makedirs(output_flow_img_folder,exist_ok=True)
    output_blur_map_folder = os.path.join(output_video_folder, "blur_mag_np")
    os.makedirs(output_blur_map_folder, exist_ok=True)
    return output_blur_img_folder, output_flow_img_folder, output_blur_map_folder

def model_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--epochs',default=20, help="iter times")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("home/jthe/BME/blur-magnitude-estimator/generate_dataset/core/raft_weights/raft-sintel.pth"))
    model = model.module
    model.to(device)
    model.eval()
    return model

def main(parms):
    avg_num = parms['avg_num'] # the number of sharp img to synthesis
    half_num = (avg_num-1)//2
    input_folder = parms['input_folder']
    # loading and setting RAFT model.
    model = model_setting()

    video_list = os.listdir(input_folder)
    video_idx = 0
    for video in video_list:
        video_idx += 1
        img_folder, flow_folder, map_folder = building_out_video_folder(parms, video)
        cur_video_path = os.path.join(input_folder,video)
        file_list = sorted(os.listdir(cur_video_path))
        for idx in range(half_num,len(file_list)-half_num, avg_num):
            print(idx,"/",len(file_list),", ",video_idx,'/',len(video_list))
            # construct outptu path 
            blur_img_path = os.path.join(img_folder, file_list[idx])
            flow_img_path = os.path.join(flow_folder, file_list[idx])
            blur_map_path = os.path.join(map_folder, file_list[idx].replace('png','npy'))

            blur_img = get_avg_blur(cur_video_path, file_list[idx-half_num:idx+half_num+1])
            flur_img, blur_map = get_blur_map(model, cur_video_path, file_list[idx-half_num:idx+half_num+1])
            cv2.imwrite(blur_img_path,blur_img)
            cv2.imwrite(flow_img_path,flur_img)
            np.save(blur_map_path, blur_map)
            # cv2.imwrite(blur_map_path,blur_map)

if __name__ == '__main__':
    parms = dict()
    parms['avg_num'] = 11
    parms['input_folder'] = "disk2/jthe/datasets/GOPRO_Large_all/test"
    parms['output_folder'] = "disk2/jthe/datasets/GOPRO_blur_magnitude/test/frame"+str(parms['avg_num'])
    main(parms)
    print(max_magnitude)