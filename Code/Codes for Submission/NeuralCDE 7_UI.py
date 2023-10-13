import os
import cv2
import pickle
import warnings
import argparse
import numpy as np
import albumentations as A

from PIL import Image
import os
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad 
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import SpectralClustering

from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk

from NeuralCDE_utils import *
from Unsupervised_model import *

warnings.filterwarnings("ignore")

def getCAM(model, img_path: list, layer) -> None:
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

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--img_input_size', type = int, default = 128)
    parser.add_argument('--CAM_image_dir', type=str, default='temp_file')

    parser.add_argument('--method', type=str, default='gradcam', choices=
                            [
                            'gradcam', 'gradcam++',
                            'scorecam', 'xgradcam',
                            'ablationcam', 'eigencam',
                            'eigengradcam', 'layercam', 'fullgrad'
                            ],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                                '/ablationcam/eigencam/eigengradcam/layercam')

    arg = parser.parse_args([])

    pathsss = []
    for img in img_path:
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

        pathsss.append(cam_image)
    return pathsss

class NeuralCDE(nn.Module):
    def __init__(self) -> None:
        super(NeuralCDE, self).__init__()
        self.args = self.getArgs()

        self.cnn = torch.load(self.args.backbone_path).to(self.args.device)
        self.target_layer = self.cnn.layer4
        self.cnn.eval()

        self.cde = NeuralCDEVisual(self.args).to(self.args.device)
        self.cde.load_state_dict(torch.load(self.args.model_path).state_dict())
        self.cde.eval()

        self.transforms = A.Compose([
            A.Resize(width=128, height=128, p=1),
            A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def forward(self, video) -> torch.Tensor:
        org_frames = []
        frames = []
        cap = cv2.VideoCapture(video)
        length = cap.get(7)
        frame_freq = int(length // 64)
        i = 0
        success, data = cap.read()

        if not success:
            warnings.resetwarnings()
            warnings.warn(f'Video {video} unreadable!')
            warnings.filterwarnings('ignore')
            return
        while success:
            if i % frame_freq == 0:
                org_frames.append(data)
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                frames.append(self.transforms(image=data)['image'])
            i += 1
            success, data = cap.read()
        cap.release()

        frames = torch.stack(frames)
        z_T = self.cde(frames.unsqueeze(dim=0).to(self.args.device))
        z_T_with_t = np.array([z_T[0].cpu().detach()[i].tolist() + [i / 5] for i in range(len(z_T[0]))])

        model = SpectralClustering(n_clusters=4, random_state=42)
        yhat = model.fit_predict(z_T_with_t)
        clusters = np.unique(yhat)
        times = []
        for cluster in clusters:
            times.append(np.where(yhat == cluster)[0][0] // 5)

        times = sorted(times)
        times += [len(frames)]

        keys = []
        for i in range(len(times) - 1):
            ps = np.argmax(np.array([float((nn.Softmax()(self.cnn(frame.unsqueeze(dim=0).to(self.args.device))[0])).max().cpu().detach()) for frame in frames[times[i]: times[i + 1]]]))
            keys.append(times[i] + ps)

        times = np.array(times) / len(times)
        times = [str(round(length * times[i] / 1000, 4)) + ' s' for i in [1, 2, 3]]
        org_frames = [org_frames[i] for i in keys]

        return org_frames, int(z_T[1]), times
                
    def getArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--backbone_path', type = str, default = 'Model\\Feature_Extractor_Co_Po\\model.pkl')
        parser.add_argument('--model_path', type = str, default = 'Model\\NeuralCDE_Naive_Co_Po\model.pkl')

        parser.add_argument('--adjoint', type = bool, default = True)
        parser.add_argument('--img_input_size', type = int, default = 128)
        parser.add_argument('--img_output_size', type = int, default = 32)
        parser.add_argument('--hidden_size', type = int, default = 16)
        parser.add_argument('--output_size', type = int, default = 2)

        parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        return parser.parse_args([])



class Kvasir_APP():
    def __init__(self) -> None:
        self.cde = NeuralCDE()
        # self.un = Unsupervised_model("model_last.pth", "G:\Test\data\labeled-images\lower-gi-tract")

        self.labels = ['Colitis', 'Polyps']

        self.root = Tk()
        self.root.title('Gastrointestinal Tract Video Diagnose System')
        self.root.geometry('1500x512')

        self.filename = ''

        self.lb = Label(self.root, text='Gastrointestinal Tract Video Diagnose System', width=150, height=5, font=('Times', '20'))
        self.lb.pack() 

        select_btn = Button(self.root,
                    text='select a video from your computer',
                    command=self.set_name,
                    width=50,
                    height=5,
                    fg='#000000',
                    bg='#FFFFFF',
                    font=('Times', '20'),
                    relief=RIDGE)
        select_btn.pack()

        continue_btn = Button(self.root,
                    text='predict',
                    command=self.show_result,
                    width=25,
                    height=2,
                    fg='#000000',
                    bg='#FFFFFF',
                    font=('Times', '20'),
                    relief=RIDGE)
        continue_btn.pack()
        
        self.root.mainloop()

    def set_name(self):
        self.filename = tkinter.filedialog.askopenfilename()
        if self.filename != '':
            self.lb.config(text='The file you selected is \n' + self.filename.split(r"/")[-1])
        else:
            self.lb.config(text='You have not selected any file!')

    def show_result(self):
        if self.filename != '':
            frames, cde_pred_label, times = self.cde(self.filename)

            result = tkinter.Toplevel()
            result.title('Gastrointestinal Tract Video Diagnose System')
            result.geometry('800x900')

            os.makedirs('temp_file', exist_ok=True)
            for i in range(4):
                cv2.imwrite(f'temp_file//{i}.jpg', cv2.resize(frames[i], (128, 128)))
            
            img0 = Image.open('temp_file//0.jpg')
            img1 = Image.open('temp_file//1.jpg')
            img2 = Image.open('temp_file//2.jpg')
            img3 = Image.open('temp_file//3.jpg')

            pimg0 = ImageTk.PhotoImage(img0)
            pimg1 = ImageTk.PhotoImage(img1)
            pimg2 = ImageTk.PhotoImage(img2)
            pimg3 = ImageTk.PhotoImage(img3)
            
            modelpath3 = "model_last.pth"
            datapath3 = r"G:\Test\data\labeled-images\unsupervised"
            unsupervised = Unsupervised_model(modelpath3,datapath3)
            labelllll1 = unsupervised.predict('temp_file//0.jpg')
            labelllll2 = unsupervised.predict('temp_file//1.jpg')
            labelllll3 = unsupervised.predict('temp_file//2.jpg')
            labelllll4 = unsupervised.predict('temp_file//3.jpg')
            
            votearr = [0,0,0,0]
            votearr[labelllll1]+=1
            votearr[labelllll2]+=1
            votearr[labelllll3]+=1
            votearr[labelllll4]+=1

            votearr = np.array(votearr)
            index = votearr.argmax()
            labelllll = unsupervised.class_map[index] #string

            lb = Label(result, text='keyframes: ', width=50, height=2, font=('Times', '20'))
            lb.pack()

            canvas = tkinter.Canvas(result, width=4*img0.width + 60, height=img0.height, bg='white')
            canvas.create_image(0, 0, image=pimg0, anchor="nw")
            canvas.create_image(img0.width + 20, 0, image=pimg1, anchor="nw")
            canvas.create_image(2 * img0.width + 40, 0, image=pimg2, anchor="nw")
            canvas.create_image(3 * img0.width + 60, 0, image=pimg3, anchor="nw")

            canvas.pack()

            lb = Label(result, text='Clipping timestampes are', width=50, height=2, font=('Times', '20'))
            lb.pack()

            clipping = ''
            for clip_time in times:
                clipping += clip_time + '  '
            lb = Label(result, text=clipping, width=50, height=2, font=('Times', '20'))
            lb.pack()

            lb = Label(result, text='Diagnose Result', width=50, height=2, font=('Times', '20'))
            lb.pack()
            lb = Label(result, text=str(labelllll), width=50, height=2, font=('Times', '40'))
            lb.pack()

            seg_btn = Button(result,
                        text='see segmentation',
                        command=self.see_seg,
                        width=25,
                        height=2,
                        fg='#000000',
                        bg='#FFFFFF',
                        font=('Times', '20'),
                        relief=RIDGE)
            seg_btn.pack()

            cam_btn = Button(result,
                        text="see model's attention",
                        command=self.see_cam,
                        width=25,
                        height=2,
                        fg='#000000',
                        bg='#FFFFFF',
                        font=('Times', '20'),
                        relief=RIDGE)
            cam_btn.pack()

            result.mainloop()

        else:
            return

    def see_seg(self):
        # return
        seg = tkinter.Toplevel()
        seg.title('Gastrointestinal Tract Video Diagnose System')
        seg.geometry('700x200')

        model = torch.load("unet3plus_v1.pth")

        img_np = []
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(4):
            img = Image.open('temp_file//' + str(i) + '.jpg')
            img = transform(img)
            img = img.to(device)
            pred = model(img.unsqueeze(0))
            pred[pred<0.5] = 0; pred[pred>=0.5] = 1
            img_np.append(pred)

        for i in range(4):
            image = img_np[i][0][0].cpu().detach().numpy().astype('uint8')
            image = cv2.resize(image, (128, 128))
            cv2.imwrite(f'temp_file//seg_{i}.jpg', 255*image)

        img0 = Image.open('temp_file//seg_0.jpg')
        img1 = Image.open('temp_file//seg_1.jpg')
        img2 = Image.open('temp_file//seg_2.jpg')
        img3 = Image.open('temp_file//seg_3.jpg')

        pimg0 = ImageTk.PhotoImage(img0)
        pimg1 = ImageTk.PhotoImage(img1)
        pimg2 = ImageTk.PhotoImage(img2)
        pimg3 = ImageTk.PhotoImage(img3)

        lb = Label(seg, text='Segmentation: ', width=50, height=2, font=('Times', '20'))
        lb.pack()

        seg_canvas = tkinter.Canvas(seg, width=4*img0.width + 60, height=img0.height, bg='white')
        seg_canvas.create_image(0, 0, image=pimg0, anchor="nw")
        seg_canvas.create_image(img0.width + 20, 0, image=pimg1, anchor="nw")
        seg_canvas.create_image(2 * img0.width + 40, 0, image=pimg2, anchor="nw")
        seg_canvas.create_image(3 * img0.width + 60, 0, image=pimg3, anchor="nw")

        seg_canvas.pack()

        seg.mainloop()

    def see_cam(self):
        cam = tkinter.Toplevel()
        cam.title('Gastrointestinal Tract Video Diagnose System')
        cam.geometry('700x200')

        model = torch.load('Model\\NeuralCDE_Naive_Co_Po\\model.pkl').feature_extra.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        img_np = getCAM(model, [f'temp_file//{i}.jpg' for i in range(4)], model.layer4)
        for i in range(4):
            cv2.imwrite(f'temp_file//cam_{i}.jpg', cv2.resize(img_np[i], (128, 128)))

        img0 = Image.open('temp_file//cam_0.jpg')
        img1 = Image.open('temp_file//cam_1.jpg')
        img2 = Image.open('temp_file//cam_2.jpg')
        img3 = Image.open('temp_file//cam_3.jpg')

        pimg0 = ImageTk.PhotoImage(img0)
        pimg1 = ImageTk.PhotoImage(img1)
        pimg2 = ImageTk.PhotoImage(img2)
        pimg3 = ImageTk.PhotoImage(img3)

        lb = Label(cam, text='Grad CAM: ', width=50, height=2, font=('Times', '20'))
        lb.pack()

        cam_canvas = tkinter.Canvas(cam, width=4*img0.width + 60, height=img0.height, bg='white')
        cam_canvas.create_image(0, 0, image=pimg0, anchor="nw")
        cam_canvas.create_image(img0.width + 20, 0, image=pimg1, anchor="nw")
        cam_canvas.create_image(2 * img0.width + 40, 0, image=pimg2, anchor="nw")
        cam_canvas.create_image(3 * img0.width + 60, 0, image=pimg3, anchor="nw")

        cam_canvas.pack()

        cam.mainloop()

if __name__ == '__main__':
    Kvasir_APP()