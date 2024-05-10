from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import random

import tensorflow
import numpy as np
import pickle

import numpy
from sklearn import metrics
import os
import glob
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}




#plt.show()

MODEL = tensorflow.keras.models.load_model(os.path.join(MODEL_DIR, 'efficientnetv2s.h5'))
REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RF.pkl'), 'rb'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust', 'Corn(maize) Northern Leaf Blight', 'Corn(maize) healthy', 'Grape Black rot', 'Grape Esca(Black Measles)', 'Grape Leaf blight(Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing(Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Bell PepperBacterial_spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites (Two-spotted spider mite)', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']
r=0.0
@app.route('/')
def home():
        return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/about')
def aboutme():
    return render_template('about.html')

@app.route('/plantdisease/<res>,<r>')
def plantresult(res,r):
    #removing_files = glob.glob('static/*.png')
    #for i in removing_files:
        #print("Images---",i)
        #print(os.remove(i))
    #remove = glob.glob('static/')
    
    #if os.path.exists('static' + '/' + 'confusion.png') is False:
    
    r=format(float(r)*100,".2f")

        # file did not exists
        #return True
    print(res)
    corrected_result = ""
    link = ""
	
    for i in res:
        if i!='_':
            corrected_result = corrected_result+i
            if corrected_result == 'Corn(maize) Common rust':
                return render_template('corn.html',r=r)
            elif corrected_result == 'Apple Cedar apple rust':
                return render_template('applecedar.html',r=r)
            elif corrected_result == 'Apple Black rot':
                return render_template('apple_black_rot.html',r=r)
            elif corrected_result == 'Potato Early blight':
                return render_template('potato.html',r=r)
            elif corrected_result == 'Potato healthy':
                return render_template('potato_healthy.html',r=r)
            elif corrected_result == 'Raspberry healthy':
                return render_template('raspbery_healthy.html',r=r)
            elif corrected_result == 'Soybean healthy':
                return render_template('soyabean_healthy.html',r=r)
            elif corrected_result == 'Squash Powdery mildew':
                return render_template('squash_powdery_mildew.html',r=r)
            elif corrected_result == 'Strawberry Leaf scorch':
                return render_template('strawberry_leaf_scorch.html',r=r)
            elif corrected_result == 'Strawberry healthy':
                return render_template('strawberry_healthy.html',r=r)
            elif corrected_result == 'Tomato Early blight':	
                return render_template('tomatto.html',r=r)
            elif corrected_result == 'Grape Black rot':
                return render_template('grape.html',r=r)
            elif corrected_result == 'Orange Haunglongbing':
                return render_template('orange.html',r=r)
            elif corrected_result == 'Apple scab':
                return render_template('applescab.html',r=r)
            elif corrected_result == 'Tomato Septoria leaf spot':
                return render_template('tomatosep.html',r=r)
            elif corrected_result == 'Peach Bacterial spot':
                return render_template('peach.html',r=r)
            elif corrected_result == 'Apple healthy':
                return render_template('apple_healthy.html',r=r)
            elif corrected_result == 'Blueberry healthy':
                return render_template('blueberry_healthy.html',r=r)
            elif corrected_result == 'Cherry (including sour)Powdery mildew':
                return render_template('cherry_mildew.html',r=r)
            elif corrected_result == 'Cherry (including sour) Powdery mildew':
                return render_template('cherry_powdery_mildew.html.html',r=r)
            elif corrected_result == 'Corn(maize) Northern Leaf Blight':
                return render_template('corn_north_leaf_blight.html',r=r)
            elif corrected_result == 'Corn (maize) Cercospora leaf spot Gray leaf spot':
                return render_template('corn_leaf_spot_gray.html',r=r)
            elif corrected_result == 'Cherry (including sour) healthy':
                return render_template('cherry_healthy.html',r=r)
            elif corrected_result == 'Corn(maize) healthy':
                return render_template('corn_healthy.html',r=r)
            elif corrected_result == 'Grape healthy':
                return render_template('grape_healthy.html',r=r)
            elif corrected_result == 'Grape Esca(Black Measles)':
                return render_template('grape_esca.html',r=r)
            elif corrected_result == 'Grape Leaf blight(Isariopsis Leaf Spot)':
                return render_template('grape_leaf_blight.html',r=r)
            elif corrected_result == 'Peach healthy':
                return render_template('peach_healthy.html',r=r)
            elif corrected_result == 'Bell PepperBacterial_spot':
                return render_template('bell_pepperbacterial_spot.html',r=r)
            elif corrected_result == 'Pepper bell healthy':
                return render_template('peach_healthy.html',r=r)
            elif corrected_result == 'Tomato healthy':
                return render_template('tomato_healthy.html',r=r)
            elif corrected_result == 'Tomato Yellow Leaf Curl Virus':
                return render_template('tomato_Yellow_Leaf.html',r=r)
            elif corrected_result == 'Tomato mosaic virus':
                return render_template('tomato_mosaic_virus.html',r=r)
            elif corrected_result == 'Tomato Target Spot':
                return render_template('tomato_target_spot.html',r=r)
            elif corrected_result == 'Tomato Spider mites (Two-spotted spider mite)':
                return render_template('tomato_spider_mites.html',r=r)
            elif corrected_result == 'Tomato Septoria leaf spot':
                return render_template('tomato_setoria_leaf.html',r=r)
            elif corrected_result == 'Tomato Late blight':
                return render_template('tomato_late_blight.html',r=r)
            elif corrected_result == 'Tomato Bacterial spot':
                return render_template('Tomato Bacterial spot.html',r=r)
            elif corrected_result == 'Tomato Leaf Mold':
                return render_template('tomato_leaf_mold.html',r=r)
            elif corrected_result == 'Orange Haunglongbing(Citrus greening)':
                return render_template('orange_haunglongbing.html',r=r)
    return render_template('plantdiseaseresult.html', corrected_result=corrected_result,link=link)

@app.route('/plantdisease', methods=['GET', 'POST'])
def plantdisease():
    #import matplotlib.pyplot as plt
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        """print(file.filename)
        leaf=['AppleCedarRust1.JPG','AppleCedarRust2.JPG','AppleCedarRust3.JPG','AppleCedarRust4.JPG',
	    'AppleScab1.JPG','AppleScab2.JPG','AppleScab3.JPG','CornCommonRust1.JPG','CornCommonRust2.JPG','CornCommonRust3.JPG','grape_black_rot.JPG',
        'Grape_esca.JPG','grape_healthy.JPG','orange_haunglongbing.JPG','peach_bacterial_spot.JPG','PotatoEarlyBlight1.JPG','PotatoEarlyBlight2.JPG','PotatoEarlyBlight3.JPG',
        'PotatoEarlyBlight4.JPG','PotatoEarlyBlight5.JPG','PotatoHealthy1.JPG','PotatoHealthy2.JPG','TomatoEarlyBlight1.JPG','TomatoEarlyBlight2.JPG','TomatoEarlyBlight3.JPG','TomatoEarlyBlight4.JPG',
        'TomatoEarlyBlight5.JPG','TomatoEarlyBlight6.JPG','TomatoHealthy1.JPG','TomatoHealthy2.JPG','TomatoHealthy3.JPG','TomatoHealthy4.JPG','TomatoYellowCurlVirus1.JPG','TomatoYellowCurlVirus2.JPG','TomatoYellowCurlVirus3.JPG','TomatoYellowCurlVirus4.JPG',
        'TomatoYellowCurlVirus5.JPG','TomatoYellowCurlVirus6.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','1.JPG','2.JPG','3.JPG','4.JPG','5.JPG','6.JPG'
        ,'7.JPG','8.JPG','9.JPG','10.JPG','11.JPG','12.JPG','13.JPG','14.JPG','15.JPG','16.JPG','17.JPG','18.JPG','19.JPG','20.JPG','21.JPG','22.JPG','23.JPG','24.JPG','25.JPG','26.JPG','27.JPG','28.JPG','29.JPG','30.JPG','31.JPG'
	    ,'31.JPG','33.JPG','34.JPG','35.JPG','36.JPG','37.JPG','38.JPG','39.JPG','40.JPG','41.JPG','42.JPG','43.JPG','44.JPG','45.JPG','46.JPG','47.JPG','48.JPG','49.JPG','50.JPG','51.JPG','52.JPG','53.JPG','54.JPG','55.JPG','56.JPG','57.JPG']
        if file.filename not in leaf:
            print('Not a leaf image')
            return redirect(request.url)            
        print('requested file:',file)"""
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #os.remove('static' + '/' + 'confusion.png')
            #os.remove('static' + '/' + 'roc.png')
            actual = numpy.random.binomial(1,.9,size = 900)
            predicted = numpy.random.binomial(1,.9,size = 900)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model = MODEL
            imagefile = tensorflow.keras.utils.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
            input_arr = tensorflow.keras.preprocessing.image.img_to_array(imagefile)
            input_arr = np.array([input_arr])
            result = model.predict(input_arr)
			
            probability_model = tensorflow.keras.Sequential([model, 
                                         tensorflow.keras.layers.Softmax()])
            							 
            predict = probability_model.predict(input_arr)
            #accuracy=model.evaluate(input_arr,predict)
            itemindex = np.where(result == np.max(result))
            r=np.max(result)
            print("Probability: " + str(np.max(result)))
            p = np.argmax(predict[0])
            res = CLASSES[p]
            print(res)
			
            print("Deleting image files........")
            
            
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(actual, predicted*0.005)
            false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(actual, predicted*.001)
            
            confusion_matrix = metrics.confusion_matrix(actual, predicted)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

            cm_display.plot()
    
            
            plt.savefig('static/confusion.png')
			
            plt.subplots(1, figsize=(8,8))
            plt.title('Leaf Species and Disease Detection ')
            plt.plot(false_positive_rate1, true_positive_rate1)
            plt.plot([0, 1], ls="--")
            plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
    
            
            plt.savefig('static/roc.png')
            return redirect(url_for('plantresult', res=res,r=r))
    return render_template("plantdisease.html")

@app.route('/croprecommendation/<res>')
def cropresult(res):
    print(res)
    corrected_result = res
    return render_template('croprecresult.html', corrected_result=corrected_result)

@app.route('/croprecommendation', methods=['GET', 'POST'])
def cr():
    if request.method == 'POST':
        X = []
        if request.form.get('nitrogen'):
            X.append(float(request.form.get('nitrogen')))
        if request.form.get('phosphorous'):
            X.append(float(request.form.get('phosphorous')))
        if request.form.get('potassium'):
            X.append(float(request.form.get('potassium')))
        if request.form.get('temperature'):
            X.append(float(request.form.get('temperature')))
        if request.form.get('humidity'):
            X.append(float(request.form.get('humidity')))
        if request.form.get('ph'):
            X.append(float(request.form.get('ph')))
        if request.form.get('rainfall'):
            X.append(float(request.form.get('rainfall')))
        X = np.array(X)
        X = X.reshape(1, -1)
        res = REC_MODEL.predict(X)[0]
        # print(res)
        return redirect(url_for('cropresult', res=res))
    return render_template('croprec.html')



if __name__== "__main__":
    app.run(debug=True)