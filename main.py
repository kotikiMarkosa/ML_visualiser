from flask import Flask, flash, redirect, render_template, \
     request, url_for
import os
import requests
import json
import time
from pics_creating import *
import time

action = None
data_type = None
filename = None
    
app = Flask(__name__, static_url_path='/static')
# app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = '#&*(#TY(#*HRIRHDO*&Y#HIUDO*#&DY*H' 



@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('start.html', action=action, data_type=data_type)


@app.route('/start/act/<pushed_button>/', methods=['GET'])
def pushed_button_funk(pushed_button):
    global action
    action = pushed_button
    print('\n\n', action, data_type, '\n\n')
  
    return render_template('start.html', action=action, data_type=data_type)


@app.route('/start/data/<data>/', methods=['GET'])
def get_data(data): 
    global data_type   
    data_type = data
    print('\n\n', action, data_type, '\n\n')
  
    return render_template('start.html', action=action, data_type=data_type)


@app.route('/start/submit/', methods=['GET'])
def submit():
    global action
    global data_type


    if action == 'clustering' and  data_type == 'data_for_clustering':
        print('\n\n CLUSTERING \n\n')
        return render_template('start.html', action=action, data_type=data_type)
    elif action == 'classification' and  data_type == 'data_for_classification':
        print('\n\n CLASSIFICATION \n\n')
        return render_template('classification.html', filename=None, data_type=data_type)
    elif action == 'regression' and  data_type == 'data_for_regression':
        print('\n\n REGRESSION \n\n')
        return render_template('start.html', action=action, data_type=data_type)
    else:
        error = 'Not classical situation'
        
    return render_template('start.html', error=error)
  
@app.route('/classification/apply/', methods=['GET', 'POST'])
def classification_methods():
    global data_type
    global filename
    names = []
    if request.method == 'POST':
        names = request.form.getlist('clf')
    if 'all' in names:
        names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes']

    
    dscreator = DSCreator(random_state=1, data_type=data_type)
    datasets = dscreator.create_ds()
    methcreator = MethodCreator(problem='classification')
    clf_dict = methcreator.create_dict()
    methods_list = [clf_dict[x] for x in names]

    clf_vis = ClfVisualiser(ds_list=datasets, methods_list=methods_list, names=names)
    filename = clf_vis.plot_comparison()
    
    print(filename)
    # time.sleep(20)
    return render_template('classification.html', filename=filename, data_type=data_type)



 

if __name__ == "__main__":
    app.run()