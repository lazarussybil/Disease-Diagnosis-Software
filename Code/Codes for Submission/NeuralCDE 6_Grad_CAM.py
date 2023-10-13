import os
import cv2
import random
import sklearn
import warnings
import argparse
import numpy as np
import seaborn as sns

from tqdm import tqdm

import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad 
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

random.seed(42)
np.random.seed(42)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=UserWarning)
sklearn.set_config(print_changed_only=True)
sns.set_style("white")

def getCAM(model, img_path: list, layer, arg) -> None:
    methods =  {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }
    target_layers = [layer]

    for img in tqdm(img_path):
        save_name = img.split('\\')[-1][:-4]

        org_img = cv2.imread(img)
        rgb_img = np.float32(cv2.resize(cv2.imread(img, 1)[:, :, ::-1], (arg.img_input_size, arg.img_input_size))) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(arg.device)
        cam_algorithm = methods[arg.method]

        with cam_algorithm(model=model, target_layers=target_layers, use_cuda=arg.use_cuda) as cam:
            cam.batch_size = 64
            grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=arg.use_cuda)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        os.makedirs(arg.CAM_image_dir, exist_ok=True)
        cv2.imwrite(os.path.join(arg.CAM_image_dir, f'{save_name}_{arg.method}_cam.jpg'), cam_image)
        cv2.imwrite(os.path.join(arg.CAM_image_dir, f'{save_name}_org.jpg'), cv2.resize(org_img, (arg.img_input_size, arg.img_input_size)))



parser = argparse.ArgumentParser()

parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--use-cuda', type=bool, default=True)
parser.add_argument('--img_input_size', type = int, default = 128)
parser.add_argument('--CAM_image_dir', type=str, default='Prediction_Visualization\\CAM_figures')
parser.add_argument('--model_path', type=str, default='Model\\NeuralCDE_Naive_Co_Po\\model.pkl')
parser.add_argument('--image_dir', default=[
    'Data\\Video_Images_Co_Po\\Train\\colitis',
    'Data\\Video_Images_Co_Po\\Train\\polyps'
])
parser.add_argument('--method', type=str, default='gradcam', choices=
                        [
                        'gradcam', 'gradcam++',
                        'scorecam', 'xgradcam',
                        'ablationcam', 'eigencam',
                        'eigengradcam', 'layercam', 'fullgrad'
                        ],
                    help='Can be gradcam/gradcam++/scorecam/xgradcam'
                            '/ablationcam/eigencam/eigengradcam/layercam')

args = parser.parse_args([])

model = torch.load(args.model_path).feature_extra.to(args.device)
paths = []
for dir in args.image_dir:
    paths += [os.path.join(dir, path) for path in os.listdir(dir)]
getCAM(model, paths, model.layer4, args)