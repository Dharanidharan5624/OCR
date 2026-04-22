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
import logging
import pandas as pd
import re

# Configure logging
logging.basicConfig(filename='log/error.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('########################################################')

def get_creation_timestamp(folder):
    return os.stat(folder).st_ctime

# Define a function to extract the number from the subfolder name
def extract_number(subfolder_name):
    match = re.search(r'\d+', subfolder_name)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number if no number is found
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
    return label[0],confidence

# Function to perform object detection on an image
def perform_detection(model, image_path,subfolder_name,image_count,ocr_missing_count,threshold=0.8):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create a folder based on the current date
    folder_name = current_date
    crops_folder_name=current_date

    folder_name = "images/results/" + folder_name
    crops_folder_name= "images/crops/"+crops_folder_name
    wrong_image_folder="images/wrong/"+current_date
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(wrong_image_folder,exist_ok=True)

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
            if len(boxes)>1:
                print(f"Error : No of Bounding box:{len(boxes)}, image path:{image_path}")
                logging.error(f"No of Bounding box:{len(boxes)}, image path:{image_path}")
                image_path = os.path.basename(image_path)
                wrong_image_folder=wrong_image_folder+"/"+image_path
                cv2.imwrite(wrong_image_folder,image)

            try:
                x1, y1, x2, y2 = np.array(boxes.xyxy.cpu()).squeeze()
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                im = image[int(y1):int(y2), int(x1):int(x2)]
                image_path = os.path.basename(image_path)
                path = crops_folder_name+"/" + image_path
                cv2.imwrite(path, im)
                image_count+=1
                ocr_result,cofidence = ocr_text(model_ocr, im, model_args, loaded_tokenizer)
                # print(ocr_result,cofidence)
                # Convert tensor to numpy array
                tensor_array = cofidence[0].numpy()

                # Check if any value is less than 0.8
                if np.any(tensor_array < 0.8):
                    ocr_result=0
                    ocr_missing_count+=1
                    print(f"Error : No of Bounding box:{len(boxes)}, image path:{image_path}")
                    logging.error(f"OCR wrong {(tensor_array)}, image path:{image_path}")


                else:
                    print("All values are greater than or equal to 0.9")

                # Define the text and its properties
                # text =str(ocr_result)
                # position = (100,100)  # Position where the text will be drawn
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 3
                # color = (0, 0, 255)  # Color in BGR format (white in this case)
                # thickness = 2
                # Draw the text on the image
                # cv2.putText(image, text, position, font, font_scale, color, thickness)
                # path2= folder_name +"/"+ image_path
                # cv2.imwrite(path2, image)

                cls = boxes.cls.tolist()  # Convert tensor to list
                conf = boxes.conf
                conf = conf.detach().cpu().numpy()
                for class_index in cls:
                    class_name = class_names[int(class_index)]
                    detections.append((subfolder_name,ocr_result,image_path))
            except Exception as e:
                print(e)
        # else:
        #     print(f"Error : No of Bounding box:{len(boxes)}, image path:{image_path}")
        #     logging.error(f"No of Bounding box:{len(boxes)}, image path:{image_path}")
        #     image_path = os.path.basename(image_path)
        #     wrong_image_folder = wrong_image_folder + "/" + image_path
        #     cv2.imwrite(wrong_image_folder, image)

    # cv2.imshow("image", cv2.resize(image, (640, 640)))
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    return detections,image_count,ocr_missing_count
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
subfolder_count=0
image_count_per_folder = {}
ocr_missing_count_set = {}

# Define the parent folder containing subfolders with images
current_date = datetime.now()
formatted_date = current_date.strftime("%Y_%m_%d")
parent_folder_path = "gc_pandora"
csv_filename = "log/"+str(formatted_date) + ".csv"
# Iterate over all subfolders and their files
# Get a list of all subfolders
subfolders = [os.path.join(parent_folder_path, name) for name in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, name))]

# Sort subfolders by the number extracted from their names
subfolders.sort(key=extract_number)
print(subfolders)

# Iterate over sorted subfolders
for subfolder in subfolders:
    # Process images within each subfolder
    print("Processing subfolder:", subfolder)
    for root, dirs, files in os.walk(subfolder):
        image_count = 0
        subfolder_count += 1
        ocr_missing_count = 0
        subfolder_name = os.path.basename(root)  # Get the name of the subfolder
        fir = True

        for filename in files:
            if filename.endswith(".JPG") or filename.endswith(".jpg"):
                # image_count += 1
                image_path = os.path.join(root, filename)
                subfolder_name = os.path.basename(root)  # Get the name of the subfolder
                print("Processing image:", filename, "from subfolder:", subfolder_name)
                # Now you can perform detection on each image_path
                detections, image_count, ocr_missing_count = perform_detection(model, image_path, subfolder_name,
                                                                               image_count, ocr_missing_count)
                all_detections.extend(detections)

        # Write detections for the current subfolder to CSV...
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    ['Folder name', 'ocr_result', 'Image Path'])
                csv_writer.writerows(all_detections)
        else:
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(all_detections)

        # Add an empty row after each subfolder's data...
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([])  # Write an empty row

        all_detections.clear()

        image_count_per_folder[subfolder_name] = image_count
        ocr_missing_count_set[subfolder_name] = ocr_missing_count

try:

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filename)

    # Sort the DataFrame by OCR result in ascending order
    df_sorted = df.dropna(subset=['ocr_result']).sort_values(by='ocr_result')

    # Group the sorted DataFrame by folder name
    grouped = df_sorted.groupby('Folder name')

    # Write the sorted results to the CSV file, folder-wise
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder name', 'ocr_result', 'Image Path'])
        for folder_name, group_df in grouped:
            group_df.to_csv(csvfile, mode='a', index=False, header=False)
            # Write an empty row after each folder's data
            csv_writer.writerow([])

except Exception as e:
    print(e)

print("Total folders:", subfolder_count)
with open(csv_filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["",""])
    csv_writer.writerow(["Total folders","",subfolder_count])

first = True
for folder, count in image_count_per_folder.items():
    # if first:
    #     first = False
    #     continue
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([folder,"",count])

    print(f"Images processed in folder '{folder}': {count}")

with open(csv_filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["",""])

# sec=True
for folder, count in ocr_missing_count_set.items():
    # if sec:
    #     sec = False
    #     continue
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([folder,"" ,"ocr missing:"+str(count)])
    print(f"Images processed in folder '{folder}': ocr missing:{count}")


# Perform detection on images...


df = pd.read_csv(csv_filename)
# Initialize a dictionary to store unique lengths and their occurrences
unique_lengths = {}

# Iterate through the ocr_result column
for index, row in df.iterrows():
    ocr_result = row['ocr_result']
    if not pd.isna(ocr_result):  # Ignore NaN values
        length = len(str(ocr_result))
        if length not in unique_lengths:
            unique_lengths[length] = {'count': 1, 'paths': [(row['Folder name'], row['Image Path'])]}
        else:
            unique_lengths[length]['count'] += 1
            unique_lengths[length]['paths'].append((row['Folder name'], row['Image Path']))

# Print unique lengths with their corresponding image path and folder name
for length, info in unique_lengths.items():
    if info['count'] <= 5:  # Ignore lengths with more than 5 occurrences
        # print(f"Unique Length: {length}")
        for folder_name, image_path in info['paths']:
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([folder_name, "", "image path:" + str(image_path)])
            print(f"  Folder Name: {folder_name}, Image Path: {image_path}")
            logging.error(f"  Folder Name: {folder_name}, Image Path: {image_path}")


