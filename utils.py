import pymupdf
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv
import os
import time
import tensorflow as tf
import tensorflow_hub as hub
from field_fn import *
from config import *

from weasyprint import HTML
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

os.add_dll_directory(GTK_FOLDER)


# insert the GTK3 Runtime folder at the beginning. Can be bin or lib, depending on path you choose while installing.
#GTK_FOLDER = r'C:\Program Files\GTK3-Runtime Win64\bin'
os.environ['PATH'] = GTK_FOLDER + os.pathsep + os.environ.get('PATH', '')

class OCR_preprocess:
    def __init__(self, pdf_path , output_dir_img):
        self.pdf_path = pdf_path
        self.srgan_model = hub.load(SAVED_MODEL_PATH)
        self.output_dir = output_dir_img

    def pdf_to_images(self):
        # Open the PDF file
        pdf_document = pymupdf.open(self.pdf_path)
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    


    def preprocess_image(self, image, pagenum):
        # Convert image to numpy array
        img = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        self.save_image(gray, pagenum)

        hr_image = tf.image.decode_image(tf.io.read_file(os.path.join(self.output_dir, f'preprocessed_page_{pagenum + 1}.png')))
        # If PNG, remove the alpha channl. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        hr_image = tf.expand_dims(hr_image, 0)
        hr_image_rgb = tf.tile(hr_image, [1, 1, 1, 3])

        return hr_image_rgb  


   
    def save_image(self, image, page_num , super_resolutionimage = False):
        # Save the preprocessed image
        if not super_resolutionimage:
            output_path = os.path.join(self.output_dir, f'preprocessed_page_{page_num + 1}.png')
            Image.fromarray(image).save(output_path)
            print(f'Saved preprocessed image: {output_path}')

        if not isinstance(image, Image.Image) and super_resolutionimage:
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
            image.save(os.path.join(self.output_dir,  "%s.jpg" % f"superres_page_{page_num + 1}"))

    def extract_pages_from_pdf(self):
        # Convert PDF to images
        images = self.pdf_to_images()        
        # Preprocess images and perform OCR
        for pagenum,img in enumerate(images):
            preprocessed_img = self.preprocess_image(img, pagenum)


            fake_image = self.srgan_model(preprocessed_img)
            fake_image = tf.squeeze(fake_image)

            # Plotting Super Resolution Image
            self.save_image(tf.squeeze(fake_image), pagenum , super_resolutionimage = True)
        
        return len(images)
            


class OCR_mainprocess:
    def __init__(self, num_pages , pdf_path = None):
        self.ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        self.num_pages = num_pages


    def extract_words_within_bbox(self , json_op, xin1, yin1, xin2, yin2):

        words_within_bbox = []
        words_within_bb_geo = {}
        confidences = []

        # Extract words within the bounding box
        for page in json_op['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        x1, y1 = word['geometry'][0]
                        x2, y2 = word['geometry'][1]

                        if (xin1 <= x1 <= xin2 and yin1 <= y1 <= yin2) or (xin1 <= x2 <= xin2 and yin1 <= y2 <= yin2):
                            words_within_bbox.append(word['value'])
                            words_within_bb_geo[word["value"]] = x1
                            confidences.append(word['confidence'])

        result_string = ' '.join(words_within_bbox)
        return result_string , np.mean(confidences), words_within_bb_geo
    
    def find_variation(self, keyword, occ , json_output_normal):
        if keyword == "T4" and occ[0] < 0.5 :
            return "var2"
        elif keyword == "T4" and occ[0] > 0.5 :
            if self.find_occurences("16A" , json_output_normal) or self.find_occurences("17A" , json_output_normal) :
              return "var3"
            else:
                return "var1"
            
        if keyword == "RL-1" and occ[1] > 0.15 :
            return "var1"
        elif keyword == "RL-1" and occ[1] < 0.15 :
            return "var2" 

        
    
    def find_occurences(self, keyword , json_output):

        occurrences = []

        for page in json_output['pages']:
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    for word in line.get('words', []):
                        if word['value'] == keyword:
                            occurrences.append(word['geometry'])

        return occurrences
    
    def get_word(self , keyword, key, json_op , x1,y1,x2,y2):

        result_norm , confidence_norm , geometry_dict= self.extract_words_within_bbox(json_op, x1, y1, x2, y2)
        result_norm = function_check[keyword][key](result_norm, geometry_dict)

        return result_norm , confidence_norm 
    
    def extract_fields(self):
        final_result_list = []
        for i in (range(0,self.num_pages,1)):

            #ht_normal , wd_normal, _ = cv2.imread(fr"C:\Users\skr25\OneDrive\Desktop\freelancer\OCR_Work\preprocessed_page_{i+1}.png").shape
            #ht_superres , wd_superres, _ = cv2.imread(fr"C:\Users\skr25\OneDrive\Desktop\freelancer\OCR_Work\superres_page_{i+1}.jpg").shape

            img_normal = DocumentFile.from_images(os.path.join(OUTPUT_DIR_IMG , fr"preprocessed_page_{i+1}.png" ))
            img_superres = DocumentFile.from_images(os.path.join(OUTPUT_DIR_IMG , fr"superres_page_{i+1}.png" ))

            result_normal = self.ocr_model(img_normal)
            result_superres = self.ocr_model(img_superres)

            json_output_normal = result_normal.export()
            json_output_superres = result_superres.export()


            for _keyword in KEYWORD:

                occurences = self.find_occurences(_keyword , json_output_normal)

                keys = field_map[_keyword]["var1"].keys()

                for occ in occurences:
                    result_dict = {}
                    result_dict["type"] = _keyword
                    occ_1, occ_2  = occ
                    occ_x1, occ_y1 = occ_1

                    variation = self.find_variation(_keyword, occ_1 , json_output_normal)

                    width = VAR[_keyword][variation][0]
                    height = VAR[_keyword][variation][1]


                    
                    for _key in keys:
                        
                        print("\n")

                        if "_value" in _key:
                            continue

                        result = ""

                        x1,y1,x2,y2 = (occ_x1 * width + field_map[_keyword][variation][_key]["x1"]) / width , (occ_y1 * height + field_map[_keyword][variation][_key]["y1"])/height \
                            , (occ_x1 * width + field_map[_keyword][variation][_key]["x2"] )/width , (occ_y1 * height + field_map[_keyword][variation][_key]["y2"])/height

                        result_norm, confidence_norm = self.get_word(_keyword, _key, json_output_normal , x1,y1,x2,y2)
                        
                        result_superres, confidence_superres = self.get_word(_keyword, _key, json_output_superres , x1,y1,x2,y2)


                        print("\n final result \n")

                        if (result_superres == "" and result_norm != ""  ):
                            #print(result_norm)
                            result = result_norm
                            
                        elif(result_superres != "" and  result_norm == "" ):
                            #print(result_superres)
                            result = result_superres

                        elif ("." in result_norm and "." in result_superres and len(result_norm) > len(result_superres)):
                            #print(result_norm)
                            result = result_norm

                        elif ("." in result_norm and "." in result_superres and len(result_norm) < len(result_superres)):
                            #print(result_superres)
                            result = result_superres

                        elif( confidence_norm > confidence_superres) :
                            #print(result_norm)
                            result = result_norm
                        else:
                            #print(result_superres)
                            result = result_superres

                        if "_key" in _key and result != "" :
                            _key = "custom_value_" + _key[-1]
                            x1,y1,x2,y2 = (occ_x1 * width + field_map[_keyword][variation][_key]["x1"]) / width , (occ_y1 * height + field_map[_keyword][variation][_key]["y1"])/height \
                                , (occ_x1 * width + field_map[_keyword][variation][_key]["x2"] )/width , (occ_y1 * height + field_map[_keyword][variation][_key]["y2"])/height

                            result_norm, confidence_norm = self.get_word(_keyword, _key ,json_output_normal , x1,y1,x2,y2)
                            result_superres, confidence_superres = self.get_word(_keyword, _key,  json_output_superres , x1,y1,x2,y2)


                            _key = result

                            if confidence_norm > confidence_superres :
                                result = result_norm
                            else:
                                result = result_superres


                        if "_key" not in _key: 
                            result_dict[_key] = result
        
                    final_result_list.append(result_dict)
        return final_result_list            

                

def write_dicts_to_csv(list_of_dicts, filename):
        max_rows = max(len(d) for d in list_of_dicts)
        
        # Prepare the data for writing
        rows = []
        for i in range(max_rows):
            row = []
            for d in list_of_dicts:
                keys = list(d.keys())
                values = list(d.values())
                if i < len(d):
                    row.append(keys[i])  
                    row.append(values[i])  
                else:
                    row.append('')  
                    row.append('')  
            rows.append(row)
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
