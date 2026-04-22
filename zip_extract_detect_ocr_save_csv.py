import os
import numpy as np
import csv
import pickle
import torch
from torchvision.transforms import ToTensor,Resize
from PIL import Image
from ultralytics import YOLO
import cv2
from datetime import datetime
def perform_inference(model, image, model_args):
    transform = ToTensor()
    resize = resize = Resize((32, 128))   # Resize the image to match the model's input size
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = resize(image)  # Resize the image
    image_tensor = transform(image).unsqueeze(0).to(model_args['device'])
    #The image is transformed to a PyTorch tensor, unsqueezed to add a batch dimension (assuming the model expects a batch of images), and then moved to the specified device.
    with torch.no_grad():
        #: This is used to disable gradient computation during inference, which helps reduce memory usage and speeds up computations.
        output = model(image_tensor)
        # Process the output as needed

    return output

def ocr_text(model,image, model_args,loaded_tokenizer):
    inference_result = perform_inference(model, image, model_args)
    # Greedy decoding
    pred = inference_result.softmax(-1)
    label, confidence = loaded_tokenizer.decode(pred)
    # return (label[0],["{:.2%}".format(value) for value in confidence[0].tolist()[:-1]])
    return label[0]

# Function to perform object detection on an image
def perform_detection(model, image_path,subfolder_name,threshold=0.8):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create a folder based on the current date
    folder_name = current_date
    crops_folder_name=current_date

    folder_name = "images/results/" + folder_name
    crops_folder_name= "images/crops/"+crops_folder_name
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(crops_folder_name,exist_ok=True)
    # Perform prediction
    try:
        results = model.predict(image, device="cpu", classes=0, conf=threshold, imgsz=640)
    except TypeError:
        results = model.predict(image, device="cpu", conf=threshold, imgsz=640)
    detections = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        if len(boxes) > 0:
            try:
                x1, y1, x2, y2 = np.array(boxes.xyxy.cpu()).squeeze()
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                im = image[int(y1):int(y2), int(x1):int(x2)]
                image_path = os.path.basename(image_path)
                actual_value=image_path.replace(".JPG","")
                path = crops_folder_name+"/" + image_path
                print(path)
                cv2.imwrite(path, im)
                ocr_result = ocr_text(model_ocr, im, model_args, loaded_tokenizer)
                # Define the text and its properties
                text =str(ocr_result)
                position = (100,100)  # Position where the text will be drawn
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                color = (0, 0, 255)  # Color in BGR format (white in this case)
                thickness = 2
                # Draw the text on the image
                cv2.putText(image, text, position, font, font_scale, color, thickness)
                path2= folder_name +"/"+ image_path
                print(path2)

                cv2.imwrite(path2, image)
                cls = boxes.cls.tolist()  # Convert tensor to list
                conf = boxes.conf
                conf = conf.detach().cpu().numpy()
                result=False
                if actual_value==ocr_result:
                    result=True
                for class_index in cls:
                    class_name = class_names[int(class_index)]
                    detections.append((subfolder_name,image_path, len(boxes), class_name, conf[0],ocr_result,result))
            except Exception as e:
                print(e)
        else:
            image_path = os.path.basename(image_path)
            detections.append((subfolder_name,image_path, 0, "", 0,0,0))
    cv2.imshow("image", cv2.resize(image, (640, 640)))
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    return detections
#load ocr model

# Define the path to the checkpoint
checkpoint_path = 'weights/ocr.ckpt'
# Define the model parameters
model_args = {
    'data_root': 'data',
    'batch_size': 1,  # Set batch size to 1 for inference on singular images
    'num_workers': 4,
    'cased': False,
    'punctuation': False,
    'new': False,  # Set to True if you want to evaluate on new benchmark datasets
    'rotation': 0,
    'device': 'cpu'  # Use 'cuda' or 'cpu' depending on your environment
}
# Load the model checkpoint
#model_ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True,map_location=torch.device('cpu'))  # Example: Replace with your model loading code
with open('tokenizer/tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)
model_ocr = torch.jit.load('pretrained_models/Pretrained.pth').eval().to('cpu')
#, if you want to save and load the complete model, including its architecture and parameters, you might use ".pt". If you only want to save and load the model parameters, you might use ".pth". The choice between them depends on your specific use case and whether you need to preserve the model architecture.
model_ocr.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict'])
model_ocr.eval()
#In PyTorch, the eval() method is used to set the model in evaluation mode. When a model is in evaluation mode, it behaves differently than during training. The primary purpose of using eval() is to disable certain operations like dropout and batch normalization during inference or evaluation.
model_ocr.to(model_args['device'])
#end ocr model

# Load the YOLO model
model_path = "weights/best.pt"
model = YOLO(model_path)
# Define class names
class_names = ['coin_id']
# Perform detection on all images in the folder
all_detections = []

# Define the parent folder containing subfolders with images
parent_folder_path = "gc_pandora"
# Iterate over all subfolders and their files
for root, dirs, files in os.walk(parent_folder_path):
    for filename in files:
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            image_path = os.path.join(root, filename)
            subfolder_name = os.path.basename(root)  # Get the name of the subfolder
            print("Processing image:", filename, "from subfolder:", subfolder_name)
            # Now you can perform detection on each image_path
            detections = perform_detection(model, image_path,subfolder_name)
            all_detections.extend(detections)

current_date = datetime.now()
formatted_date = current_date.strftime("%Y_%m_%d")
csv_filename = "log/"+str(formatted_date) + ".csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder name','Image Path', 'Number of Boxes', 'Class Name', 'Confidence', 'ocr_result', 'result',])
        csv_writer.writerows(all_detections)
else:
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_detections)
# Perform detection on images...
print("Detection results have been saved to", csv_filename)


