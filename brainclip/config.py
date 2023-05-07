data_folder = "/datadrive_m2/alice/brain-CLIP/data/"
# raw data paths
raw_data_base = data_folder+"raw/"
raw_images = raw_data_base+"images/Alice_IN-BodyScanData-03/"
raw_images_ext = raw_data_base+"images/IN-BodyScanData-03-Alice-265"
raw_text = raw_data_base+"text/"
parsed_reports = raw_text+"parsed_reports.csv"
parsed_reports_ext = raw_text+"parsed_reports_ext.csv"
# model splits
train_data = data_folder+"train/"
valid_data = data_folder+"valid/"
test_data = data_folder+"test/"
# trained model path
model_folder = "/datadrive_m2/alice/brain-CLIP/brainclip/model/"
experiments_folder = model_folder+"experiments/"
final_model_path = experiments_folder+"brainclip_final.pt"
classification_model_path = experiments_folder+"brainclip_class_final.pt"