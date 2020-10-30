import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pymongo import MongoClient
import time
from scipy.spatial import distance as dist
import pandas as pd

print(cv2.__version__)

settings = {
            "MongoDB": {
                        "IP": "mongodb://localhost:27017/",
                        "port": "",
                        "create_new": False,
                        "database": "assignment_database", #assignment_database
                        "collection": "all_images" #all_images
                        },
            "Images_Folder": {
                        "path": ".\\Assignment_Images",
                        "tag": "headphones"
                        },
            "Test_Image": {
                        "path": ".\\Testing_Images",
                        "file": "headphone_to_compare.jpg",
                        "algorithm": "histogram", # sift, orb, surf, histogram
                        "matcher": {
                                    'method': "FLANN", #FLANN, bruteforce
                                    'ratio': 0.75
                                    },
                        "metric": "dice", # euclidean, cityblock, minkowski, 
                                               # chebyshev, kulsinski, cosine, jaccard, dice
                        "k-NN": 20
                        },
    }


class MongoDB_API:
    def __init__(self, settings):
        self.ip = settings['MongoDB']['IP']
        self.port = settings['MongoDB']['port']
        self.database_name = settings['MongoDB']['database']
        self.collection_name = settings['MongoDB']['collection']
        self.create_new = settings['MongoDB']['create_new']
        self._database = None 
        self._collection = None
        
    def connect(self):
        try:
            if self.port == '':
                client = MongoClient(self.ip)
            else:
                client = MongoClient(self.ip, int(self.port))
            if not self.create_new:
                dbnames = client.list_database_names()
                print("Databases are: ", dbnames)
                if self.database_name not in dbnames: 
                    raise ValueError("Database doesn't exist.")
            db = client[self.database_name]
            self._database = db
            collection = db[self.collection_name]
            print('Connected to MongoDB')
            self._collection = collection
            return collection
        except:
            raise Exception('Connection failed')

    def insert(self, doc):
        data = self._collection.insert_one(doc) 
        return data

    
class ImageAnalysis:
    def __init__(self, settings):
        self.images_folder = settings['Images_Folder']['path']
        self.test_image_folder = settings['Test_Image']['path']
        self.test_image = settings['Test_Image']['file']
        self.tag = settings['Images_Folder']['tag']
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._surf = cv2.xfeatures2d.SURF_create()
        self._orb = cv2.ORB_create()
        
    def read_img(self, image):
        try:
            img = cv2.imread(image)
            img = cv2.resize(img,(500,500))
        except:
            raise ValueError("Wrong type of image:", image)
        return img
    
    def encode_keypoints(self, keypoints):
        ## To store them in mongodb - mongodb compatible
        encoded_keypoints = []
        for point in keypoints:
            temp = (point.pt, 
                    point.size, 
                    point.angle, 
                    point.response, 
                    point.octave, 
                    point.class_id)
            encoded_keypoints.append(temp)
        return encoded_keypoints
    
    def decode_keypoints(self, encoded_keypoints):
        decoded_keypoints = []
        for point in encoded_keypoints:
            temp = cv2.KeyPoint(x=point[0][0], 
                                y=point[0][1],
                                _size=point[1],
                                _angle=point[2], 
                                _response=point[3],
                                _octave=point[4], 
                                _class_id=point[5]) 
            decoded_keypoints.append(temp)
        return decoded_keypoints
    
    def sift(self, image):
        kp, desc = self._sift.detectAndCompute(image, None)
        # Converting descriptors to list. Mongodb cannot encode numpy.ndarray
        encoded_kp = self.encode_keypoints(kp)
        sift = {
                "keypoints": encoded_kp,
                "descriptors": desc.tolist()
            }        
        return sift
    
    def surf(self, image):
        kp, desc = self._surf.detectAndCompute(image, None)
        encoded_kp = self.encode_keypoints(kp)
        surf = {
                "keypoints": encoded_kp,
                "descriptors": desc.tolist()
            }
        return surf
    
    def orb(self, image):
        kp, desc = self._orb.detectAndCompute(image, None)
        encoded_kp = self.encode_keypoints(kp)
        orb = {
                "keypoints": encoded_kp,
                "descriptors": desc.tolist()
            }
        return orb
    
    def histogram(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], 
                            [0, 1, 2], 
                            None, 
                            [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        # Return list type, to be supported by MongoDB
        return hist.tolist()
    
    def create_obj(self, image, algorithm='all'):
        obj = {}
        obj['image_file'] = image
        obj['tag'] = self.tag
        img = self.read_img(image)
        if algorithm == 'all':
            (SIFT, ORB, SURF, HISTOGRAM) = (True, True, True, True)
        elif algorithm == 'sift':
            (SIFT, ORB, SURF, HISTOGRAM) = (True, False, False, False)
        elif algorithm == 'orb':
            (SIFT, ORB, SURF, HISTOGRAM) = (False, True, False, False)
        elif algorithm == 'surf':
            (SIFT, ORB, SURF, HISTOGRAM) = (False, False, True, False)
        elif algorithm == 'histogram':
            (SIFT, ORB, SURF, HISTOGRAM) = (False, False, False, True)
        else:
            raise ValueError('Algorithm not supported.')
        
        ## SIFT OBJECT
        if SIFT:
            obj['sift'] = self.sift(img)
        
        ## ORB OBJECT
        if ORB:
            obj['orb'] = self.orb(img)
        
        ## SURF OBJECT
        if SURF:
            obj['surf'] = self.surf(img)
        
        ## HISTOGRAM OBJECT
        if HISTOGRAM:
            obj['histogram'] = self.histogram(img)
        
        return obj
    
    def get_images_from(self, folder_path):
        img_list = []
        for fname in os.listdir(folder_path):
            path = os.path.join(folder_path, fname)
            if not os.path.isdir(path):
                # skip directories
                img_list.append(path)
        return img_list
    

################ HELPER FUNCTIONS ################

def insert_images_to_MongoDB(settings):
    mongo = MongoDB_API(settings)
    my_collection = mongo.connect()
    analysis = ImageAnalysis(settings)
    images = analysis.get_images_from(settings['Images_Folder']['path'])
    # print(images)
    for image in images:
        print("Inserting image: ", image)
        analyzed_image = analysis.create_obj(image, algorithm='all')
        my_collection.insert_one(analyzed_image)


def sort_results(results_dictionary, num_of_results, reverse):
    sorted_dict = {k: v for k, v in sorted(results_dictionary.items(), 
                                           key=lambda item: item[1]["similarity"],
                                           reverse=reverse)}
    return list(sorted_dict.keys())[:num_of_results], sorted_dict

def draw_image_matches(img1, kp1, img2, kp2, good_points, percentage):
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_points, None)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.set_title(percentage)
    plt.imshow(result)
    plt.show()
    
def score(my_list, tag):
    counter = 0
    for entry in my_list:
        if entry == tag:
            counter +=1
    return counter

####################################################

def main(settings):
    ## Check if there already is a Database. If not, create one!
    RATIO = settings['Test_Image']['matcher']['ratio']
    if RATIO <=0 or RATIO > 1:
        raise ValueError("Ratio should be a value between 0 and 1.")
    if settings["MongoDB"]["create_new"]:
        insert_images_to_MongoDB(settings)
        time.sleep(5)
    mongo = MongoDB_API(settings)
    my_collection = mongo.connect()
    analysis = ImageAnalysis(settings)
    # image_file_to_compare = analysis.get_images_from(settings['Test_Image']['path'])
    img_to_compare_path = settings['Test_Image']['path'] + "/" + settings['Test_Image']['file']
    img_to_compare = analysis.read_img(img_to_compare_path)
    analysed_img_to_compare = analysis.create_obj(img_to_compare_path, algorithm=settings['Test_Image']['algorithm'])
    # For SIFT, ORB and SURF, the procedure is similar. For Histogram it is a bit different.
    # Thus, we have the 2 following cases:
    if settings['Test_Image']['algorithm'] == 'sift' or settings['Test_Image']['algorithm'] == 'orb' or settings['Test_Image']['algorithm'] == 'surf':
        if settings['Test_Image']['metric'] != 'euclidean':
            raise ValueError("Only euclidean distance is currently supported.")
        kp_to_compare = analysis.decode_keypoints(analysed_img_to_compare[settings['Test_Image']['algorithm']]['keypoints'])
        desc_to_compare = np.asarray(analysed_img_to_compare[settings['Test_Image']['algorithm']]['descriptors'], dtype=np.float32)
        # We need to compare the descriptors of the analysed_img_to_compare to the 
        # descriptors of all images found in our database. We can use 2 methods:
        # 1) The bruteforce method: compares the descriptors from each image one by one
        # and find the best matches. It is computationally expensive
        # 2) FLANN (Fast Library for Approximate Nearest Neighbors): much faster. Finds
        # a good matching but not necessearily the best possible one.
        if settings['Test_Image']['matcher']['method'] == 'bruteforce':
            matcher = cv2.BFMatcher()
        elif settings['Test_Image']['matcher']['method'] == 'FLANN':
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        results = {}
        for image in my_collection.find():
            img_desc = np.asarray(image[settings['Test_Image']['algorithm']]['descriptors'], dtype=np.float32)
            img_kp = analysis.decode_keypoints(image[settings['Test_Image']['algorithm']]['keypoints'])
            matches = matcher.knnMatch(desc_to_compare, img_desc, k=2)
            good_points = []
            for match1, match2 in matches:
                # IF MATCH 1 DISTANCE IS LESS THAN THE SETTINGS-RATIO OF MATCH 2 DISTANCE
                # THEN DESCRIPTOR WAS A GOOD MATCH, LETS KEEP IT
                if match1.distance < RATIO * match2.distance:
                    good_points.append(match1)
            matching_percentage = len(good_points) / min(len(kp_to_compare), len(img_kp)) * 100
            results[image["_id"]] = {
                                    "similarity":matching_percentage,
                                    "good_points": good_points,
                                    "tag": image['tag']
                                    }
                                     
        id_results_list, id_results_dictionary = sort_results(results, settings['Test_Image']['k-NN'], reverse=True)
        for _id in id_results_list:
            image_DB = my_collection.find_one({ '_id' : _id})
            image_DB_file_path = image_DB['image_file']
            image_DB_file = analysis.read_img(image_DB_file_path)
            img__DB_desc = np.asarray(image_DB[settings['Test_Image']['algorithm']]['descriptors'], dtype=np.float32)
            img_DB_kp = analysis.decode_keypoints(image_DB[settings['Test_Image']['algorithm']]['keypoints'])
            draw_image_matches(img_to_compare,
                                kp_to_compare,
                                image_DB_file,
                                img_DB_kp,
                                results[_id]['good_points'],
                                results[_id]['similarity'])
    elif settings['Test_Image']['algorithm'] == 'histogram':
        hist_to_compare = np.asarray(analysed_img_to_compare[settings['Test_Image']['algorithm']], dtype=np.float32)
        results = {}
        if settings['Test_Image']['metric'] == 'euclidean':
            metric = dist.euclidean
        elif settings['Test_Image']['metric'] == 'cityblock':
            metric = dist.cityblock
        elif settings['Test_Image']['metric'] == 'minkowski':
            metric = dist.minkowski
        elif settings['Test_Image']['metric'] == 'chebyshev':
            metric = dist.chebyshev
        elif settings['Test_Image']['metric'] == 'kulsinski':
            metric = dist.kulsinski
        elif settings['Test_Image']['metric'] == 'cosine':
            metric = dist.cosine
        elif settings['Test_Image']['metric'] == 'jaccard':
            metric = dist.jaccard   
        elif settings['Test_Image']['metric'] == 'dice':
            metric = dist.dice
        else:
            raise ValueError("Distance measure selected is not supported.")
        for image in my_collection.find():
            img_hist = np.asarray(image[settings['Test_Image']['algorithm']], dtype=np.float32)
            d = metric(hist_to_compare, img_hist)
            results[image["_id"]] = {
                                    "similarity":d,
                                    "tag": image['tag']
                                    }
        id_results_list, id_results_dictionary = sort_results(results, settings['Test_Image']['k-NN'], reverse=False)
        img_to_compare = cv2.cvtColor(img_to_compare, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize = (12,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img_to_compare)
        plt.axis("off")
        fig = plt.figure("Results: %s" % (img_to_compare_path))
        fig.suptitle(img_to_compare_path, fontsize = 5)
        for i, _id in enumerate(id_results_list):
            image_DB = my_collection.find_one({ '_id' : _id})
            image_DB_file_path = image_DB['image_file']
            image_DB_file = analysis.read_img(image_DB_file_path)
            image_DB_file = cv2.cvtColor(image_DB_file, cv2.COLOR_BGR2RGB)
            ax = fig.add_subplot(1, len(id_results_list), i + 1)
            ax.set_title(str(round(results[_id]['similarity'],2))) # + '\n'+' '+image_DB_file_path.split("\\")[-1])
            plt.imshow(image_DB_file)
            plt.axis("off")
        fig.tight_layout()
        plt.show()
    ## The following 2 lines are used for the result section. Our main
    ## method returns a dataframe with our results, so that it can be processed
    ## to extract the precision. Uncomment if you want to show the precision
    
    # df = pd.DataFrame.from_dict(id_results_dictionary, orient='index')    
    # return df
        
if __name__ == "__main__":
    # start_time = time.time()
    # uncomment the following line if you want only the results
    main(settings)
    # uncomment the following lines if you want both results and precision
    
    # df = main(settings)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # # df.to_csv('histogram_headphone_euclidean.csv')
    # final_df = df[:20]
    # my_list = final_df['tag'].tolist()
    # print(score(my_list, settings['Images_Folder']['tag']))
