import time
import numpy as np
import pandas as pd
import pickle as pkl

# Parameters. #
min_side = 512
max_side = 512
l_jitter = 512
u_jitter = 512

# Load the COCO dataset. #
tmp_pd_file = \
    "C:/Users/admin/Desktop/Data/COCO/object_boxes.csv"
raw_coco_df = pd.read_csv(tmp_pd_file)

# Remember to add 1 more class for background. #
coco_label = pd.read_csv(
    "C:/Users/admin/Desktop/Data/COCO/labels.csv")
list_label = sorted([
    coco_label.iloc[x]["name"] \
    for x in range(len(coco_label))])
label_dict = dict([(
    x, list_label[x]) for x in range(len(list_label))])
index_dict = dict([(
    list_label[x], x) for x in range(len(list_label))])

tmp_col_df  = ["filename", "img_width", "img_height", "id", 
              "x_lower", "y_lower", "box_width", "box_height"]
image_files = sorted(list(pd.unique(raw_coco_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), "images in COCO dataset.")

# Find a way to remove duplicate indices from the data. #
# Total output classes is n_classes + centerness (1) + regression (4). #
print("Formatting the object detection bounding boxes.")
start_tm = time.time()

coco_objects = []
for n_img in range(len(image_files)):
    img_file = image_files.iloc[n_img]["filename"]
        
    tmp_filter = raw_coco_df[
        raw_coco_df["filename"] == img_file]
    tmp_filter = tmp_filter[[
        "img_width", "img_height", "id", 
        "x_lower", "y_lower", "box_width", "box_height"]]
    
    n_objects  = len(tmp_filter)
    tmp_bboxes = []
    tmp_labels = []
    for n_obj in range(n_objects):
        tmp_object = tmp_filter.iloc[n_obj]
        tmp_label  = coco_label[
            coco_label["id"] == \
            tmp_object["id"]].iloc[0]["name"]
        tmp_cls_id = int(index_dict[tmp_label])
        
        img_width  = tmp_object["img_width"]
        img_height = tmp_object["img_height"]
        box_x_min  = tmp_object["x_lower"] / img_width
        box_x_max  = \
            tmp_object["x_lower"] + tmp_object["box_width"]
        box_x_max  = box_x_max / img_width
        box_y_min  = tmp_object["y_lower"] / img_height
        box_y_max  = \
            tmp_object["y_lower"] + tmp_object["box_height"]
        box_y_max  = box_y_max / img_height
        
        box_dims = min(
            tmp_object["box_width"], tmp_object["box_height"])
        if box_dims > 0:
            tmp_bbox  = np.array([
                box_x_min, box_y_min, 
                box_x_max, box_y_max])
            tmp_class = np.array(tmp_cls_id)
            
            tmp_labels.append(np.expand_dims(tmp_class, axis=0))
            tmp_bboxes.append(np.expand_dims(tmp_bbox, axis=0))
    
    if len(tmp_labels) > 0:
        n_labels = len(tmp_labels)
        tmp_labels = np.concatenate(tmp_labels, axis=0)
        tmp_labels = tmp_labels.reshape((n_labels,))
        tmp_bboxes = np.concatenate(tmp_bboxes, axis=0)
        tmp_objects = {"bbox": tmp_bboxes, 
                       "label": tmp_labels}
        
        coco_objects.append({
            "image": img_file, 
            "min_side": min_side, 
            "max_side": max_side, 
            "l_jitter": l_jitter, 
            "u_jitter": u_jitter, 
            "objects": tmp_objects})
    
    if (n_img+1) % 2500 == 0:
        elapsed_tm = (time.time() - start_tm) / 60.0
        print(str(n_img+1), "annotations processed", 
               "(" + str(elapsed_tm), "minutes).")
print("Total of", str(len(coco_objects)), "images.")

print("Saving the file.")
save_pkl_file = "C:/Users/admin/Desktop/Data/COCO/"
save_pkl_file += "coco_data_fcos.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(index_dict, tmp_save)
    pkl.dump(coco_objects, tmp_save)
print("COCO data processed.")
