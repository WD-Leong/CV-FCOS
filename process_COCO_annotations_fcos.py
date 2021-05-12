
import json
import time
import pandas as pd

# Load COCO training data. #
print("Loading training dataset.")
start_tm = time.time()

tmp_path = "C:/Users/admin/Desktop/Data/COCO/"
tmp_json = json.loads(open(
    tmp_path + "annotations/instances_train2014.json").read())

tmp_bbox  = tmp_json["annotations"]
tmp_imgs  = pd.DataFrame(tmp_json["images"])
tmp_label = pd.DataFrame(tmp_json["categories"])
tmp_label.to_csv(tmp_path + "labels.csv", index=False)

n_object = 0
tmp_list = []
for tmp_obj in tmp_bbox:
    tmp_img  = tmp_imgs[tmp_imgs["id"] == tmp_obj["image_id"]]
    tmp_box  = tmp_obj["bbox"]
    tmp_file = tmp_path + "train2014/" + tmp_img["file_name"].iloc[0]
    
    tmp_width  = tmp_img.iloc[0]["width"]
    tmp_height = tmp_img.iloc[0]["height"]
    tmp_obj_id = tmp_obj["category_id"]
    tmp_object = tmp_label[
        tmp_label["id"] == tmp_obj["category_id"]]["name"]
    tmp_list.append((tmp_file, tmp_width, tmp_height, tmp_obj_id, 
                     tmp_box[0], tmp_box[1], tmp_box[2], tmp_box[3]))
    
    n_object += 1
    if n_object % 10000 == 0:
        print(str(n_object), "objects processed.")
del tmp_json, tmp_bbox, tmp_imgs

# Load COCO validation data. #
print("Loading validation dataset.")

tmp_json = json.loads(open(
    tmp_path + "annotations/instances_val2014.json").read())

tmp_bbox  = tmp_json["annotations"]
tmp_imgs  = pd.DataFrame(tmp_json["images"])
for tmp_obj in tmp_bbox:
    tmp_img  = tmp_imgs[tmp_imgs["id"] == tmp_obj["image_id"]]
    tmp_box  = tmp_obj["bbox"]
    tmp_file = tmp_path + "val2014/" + tmp_img["file_name"].iloc[0]
    
    tmp_width  = tmp_img.iloc[0]["width"]
    tmp_height = tmp_img.iloc[0]["height"]
    tmp_obj_id = tmp_obj["category_id"]
    tmp_object = tmp_label[
        tmp_label["id"] == tmp_obj["category_id"]]["name"]
    tmp_list.append((tmp_file, tmp_width, tmp_height, tmp_obj_id, 
                     tmp_box[0], tmp_box[1], tmp_box[2], tmp_box[3]))
    
    n_object += 1
    if n_object % 10000 == 0:
        print(str(n_object), "objects processed.")
del tmp_json, tmp_bbox, tmp_imgs

tmp_col_df = ["filename", "img_width", "img_height", "id", 
              "x_lower", "y_lower", "box_width", "box_height"]
tmp_obj_df = pd.DataFrame(tmp_list, columns=tmp_col_df)
tmp_obj_df.to_csv(tmp_path + "object_boxes.csv", index=False)

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", str(len(tmp_obj_df)), "images processed.")
print("Elapsed Time:", str(elapsed_tm), "mins.")
