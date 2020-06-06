# -*- coding: utf-8 -*-
"""
@author: Saif Mathur
"""

'''
DESCRIPTION:
In this project, I`ve tried to classify players into defense or attack
based on player skill and other data,

Dataset was taken by FIFA`s official website.


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




dataset = pd.read_csv('CompleteDataset.csv')
dataset.head()

#GK pos removed

columns_req = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys', 'Preferred Positions']


columns_rearranged = ['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Long shots','Penalties', 'Shot power', 'Volleys', 
       'Short passing', 'Long passing',
       'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',
       'Strength', 'Vision', 'Acceleration', 'Agility', 
       'Reactions', 'Stamina', 'Balance', 'Ball control','Composure','Jumping', 
       'Sprint speed', 'Positioning','Preferred Positions']


new_dataset = dataset[columns_rearranged]
#new_dataset = pd.DataFrame(new_dataset)                      ] 
new_dataset.head()

#removing GK 

new_dataset['Preferred Positions'] = new_dataset['Preferred Positions'].str.strip()
new_dataset = new_dataset[new_dataset['Preferred Positions'] != 'GK']
new_dataset.head()

new_dataset.isnull().values.any()

p = new_dataset['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()








df_new = new_dataset.copy()
df_new.drop(df_new.index, inplace=True)

for i in p:
    df_temp = new_dataset[new_dataset['Preferred Positions'].str.contains(i)]
    df_temp['Preferred Positions'] = i
    df_new = df_new.append(df_temp, ignore_index=True)
    
df_new.iloc[::500, :]





cols = [col for col in new_dataset.columns if col not in ['Preferred Positions']]

for i in cols:
    df_new[i] = df_new[i].apply(lambda x: eval(x) if isinstance(x,str) else x)

df_new.iloc[::500, :]




#atk 1 defense 0

mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1,
           'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}


df_new = df_new.replace({'Preferred Positions':mapping})



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_new.iloc[:,0:29] = sc.fit_transform(df_new.iloc[:,0:29])
 


#df_new is cleaned at this part

X = df_new.iloc[:,:-1].values
y = df_new.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


from sklearn.dummy import DummyClassifier
dc = DummyClassifier(strategy = 'most_frequent')
dc.fit(X_train,y_train)

'''
from sklearn.linear_model import LogisticRegression

classifier_log = LogisticRegression().fit(X_train,y_train)

y_pred = classifier_log.predict(X_test)

acc_log = classifier_log.score(X_test,y_test)
'''


import keras
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()

classifier.add(Dense(output_dim = 15, init = 'uniform',activation = 'relu',input_dim = 29))
classifier.add(Dense(output_dim = 15, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

history = classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)

y_pred_ann = classifier.predict(X_test)
y_pred_ann = y_pred_ann > 0.5


from sklearn.metrics import confusion_matrix
cm_for_ann = confusion_matrix(y_test,y_pred_ann)

acc_ann = (4654/5451)*100



#visualization
from ann_visualizer.visualize import ann_viz;
ann_viz(classifier, title="plot for project")


#summary for classifier
from keras.utils import plot_model
plot_model(model = (y,y_pred_ann), to_file='model.png')



import matplotlib.pyplot as plt

# Plot training accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()








