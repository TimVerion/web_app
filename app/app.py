from deep_code.models import *
from deep_code.utils.utils import *
from deep_code.utils.datasets import *

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import os
import time
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
import base64


############################################################################
image_folder = "deep_code/data/samples"
model_def = "deep_code/config/yolov3.cfg"
weights_path = "deep_code/weights/yolov3.weights"
class_path = "deep_code/data/coco.names"
conf_thres = 0.8
nms_thres = 0.4
batch_size = 1
n_cpu = 0
img_size = 416
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up model
model = Darknet(model_def, img_size=img_size).to(device)
if weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))

model.eval()  # Set in evaluation mode

dataloader = DataLoader(
    ImageFolder(image_folder, img_size=img_size),
    batch_size=batch_size,
    shuffle=False)

classes = load_classes(class_path)  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def detect():
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    print("\nPerforming object detection:")
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        print(batch_i , img_paths,input_imgs.shape)
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
        # Log progress
        current_time = time.time()
        # 保存图片和定位信息
        imgs.extend(img_paths)
        img_detections.extend(detections)
    print("\nSaving samples:")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top",
                         bbox={"color": color, "pad": 0})
            # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig("static/b.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

############################################################################

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# 输出
@app.route('/')
def hello_world():
    return render_template('index.html')


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/api/upload', methods=['POST'])
def upload():
    """
    api:http://127.0.0.1:5000/api/upload
    """
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        upload_path = os.path.join(image_folder, "b.jpg")
        print("now 1 pic :",upload_path)
        f.save(upload_path)
        folder_name = f.filename  # 上传文件名字
        detect()
        return render_template('show.html')


@app.route('/cap', methods=['GET'])
def hello_cap():
    return render_template('capture.html')

@app.route('/show',methods=['GET'])
def hello_pic():
    return render_template('show.html')

@app.route('/api/capture', methods=['POST'])
def capture():
    """
    api:http://127.0.0.1:5000/api/capture
    """
    base64_1 = request.json["base64_11"]
    imgdata = base64.b64decode(base64_1)
    file = open(os.path.join(image_folder,"b.jpg"), 'wb')
    file.write(imgdata)
    file.close()
    detect()
    return jsonify(results=['ok'])


if __name__ == '__main__':
    app.run()
