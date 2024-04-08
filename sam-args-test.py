import os
import torch
import cv2
import sys
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
 
# Add the path of 'segment_anything' module to sys.path
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
 
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
# MaskGenerator Class
class MaskGenerator:
    # Initialize the class with model type, device, and checkpoint path
    def __init__(self, model_type="vit_h", device="cuda", checkpoint_path=None):
        self.model_type = model_type
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.mask_generator = None
 
    # Load the model into the specified device
    def load_model(self):
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)
 
    # Initialize the mask generator with the given parameters
    def initialize_mask_generator(self, 
                                  points_per_side, 
                                  points_per_batch, 
                                  pred_iou_thresh, 
                                  stability_score_thresh, 
                                  stability_score_offset, 
                                  box_nms_thresh, 
                                  crop_n_layers,
                                  crop_nms_thresh,
                                  crop_n_points_downscale_factor,
                                  min_mask_region_area):
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )
 
    # Generate masks, color them, and return them along with their counts
    def generate_and_return_colored_masks(self, image):
        masks = self.mask_generator.generate(image)
        combined_mask = np.zeros_like(image)

        colored_masks = []
 
        np.random.seed(seed=32)
        for mask_data in masks:
            mask = mask_data['segmentation']
            mask = mask.astype(np.uint8)
 
            random_color = np.random.randint(0, 256, size=(3,))
 
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = random_color # 将 colored_mask 中与 mask 中值为 1 的像素对应的位置着色为随机颜色

            np.clip(colored_mask, 0, 255)
            colored_masks.append(colored_mask)
 
            combined_mask += colored_mask
 
        combined_mask = np.clip(combined_mask, 0, 255)

        # 将所有的 colored_masks 按照每组最多 5 张拼接成一张图
        # 如果最后一行的图像数量不足5张，使用黑色的占位图像填充
        while len(colored_masks) % 5 != 0:
            placeholder = np.zeros_like(colored_masks[0])
            colored_masks.append(placeholder)

        grouped_masks = [colored_masks[i:i+5] for i in range(0, len(colored_masks), 5)]
        combined_colored_masks = np.concatenate([np.concatenate(group, axis=1) for group in grouped_masks], axis=0)
 
        return combined_mask,combined_colored_masks
 
# Check the existence of the checkpoint file and other specifications
def check_status():
    checkpoint_path = os.path.join("D:\AI\sam-model", "sam_vit_h_4b8939.pth")
    print(checkpoint_path, "; exist:", os.path.isfile(checkpoint_path))
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    return checkpoint_path
 
# Function to process the image and generate masks
def process_image(
        image,
        points_per_side,
        points_per_batch,
        pred_iou_thresh,
        stability_score_thresh, 
        stability_score_offset, 
        box_nms_thresh,
        crop_n_layers, 
        crop_nms_thres,
        crop_n_points_downscale_factor, 
        min_mask_region_area):
    checkpoint_path = check_status()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
 
    mask_gen = MaskGenerator(checkpoint_path=checkpoint_path)
    mask_gen.load_model()
    mask_gen.initialize_mask_generator(points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thres, crop_n_points_downscale_factor, min_mask_region_area)
    combined_mask,combined_colored_masks = mask_gen.generate_and_return_colored_masks(image)

    return combined_mask,combined_colored_masks
 
 
# Main function to run the application
if __name__ == "__main__":
    create_directory("images")
    inputs = [
            gr.Image(label="Input Image - Upload an image to be processed."), 
            gr.Slider(minimum=4, maximum=128, step=4, value=32, label="points_per_side"),  # points_per_side
            gr.Slider(minimum=4, maximum=128, step=4, value=64, label="points_per_batch"),  # points_per_batch
            gr.Slider(minimum=0, maximum=1, step=0.001, value=0.880, label="pred_iou_thresh"),  # pred_iou_thresh
            gr.Slider(minimum=0, maximum=1, step=0.001, value=0.950, label="stability_score_thresh"),  # stability_score_thresh
            gr.Slider(minimum=0, maximum=2, step=0.01, value=1.00, label="tability_score_offset"),  # tability_score_offset
            gr.Slider(minimum=0, maximum=1, step=0.01, value=0.70, label="box_nms_thresh"),  # box_nms_thresh
            gr.Slider(minimum=0, maximum=10, step=1, value=0, label="crop_n_layers"),  # crop_n_layers
            gr.Slider(minimum=0, maximum=1, step=0.01, value=0.70, label="crop_nms_thresh"),  # crop_nms_thresh
            gr.Slider(minimum=0, maximum=10, step=1, value=1, label="crop_n_points_downscale_factor"),  # crop_n_points_downscale_factor
            gr.Slider(minimum=0, maximum=500, step=1, value=100, label="min_mask_region_area"),  # min_mask_region_area
    ]
    gr.Interface(fn=process_image, inputs=inputs, 
                 outputs=[#gr.Image(type="pil",label="Original Image"),
                          #gr.Image(type="pil",label="Overlay Image"),
                          gr.Image(type="pil",label="Original mask Image"),
                          gr.Image(type="pil",label="Every colored mask Images")
                        ]
    ).launch(server_port=7888,share=True)