#continue this class from google colab ,as in visual studio there are some installation errors
import pandas as pd
import plotly.express as px




#Now let's quickly create a dataframe and use it to create a machine learning model for Decision Tree.




#Column Name
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

df = pd.read_csv("./119/diabetes.csv", names=col_names).iloc[1:]

print(df.head())
#---------------------------------------------------------------
#Select the features

features = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = df[features]
y = df.label
print(y)
#Splitting the data into training and testing and fitting it in the model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Initialising the Decision Tree Model
clf = DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(X_train,y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# accuracy:0.66
#so our model can predict if the person has diabetis with 0.66 accuracy
#------------------------------------------------------------------------

#Visualising the Decision Tree
#Now that we have have built a decision tree model, that can predict with an accuracy score of 0.67 if a person has diabetes or not based on their data, is there a way we can visualise it?


#There sure is. Let's see how!



#Fun Fact
#To create a visualisation for the Decision Tree Classifier we build above, we will use the export_graphviz module of python to first convert the data into text that we can read and understand, and then we'll use the pydotplus module to convert this text into an image.

from sklearn.tree import export_graphviz
from six import StringIO 
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO() #Where we will store the data from our decision tree classifier as text.

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1'])

print(dot_data.getvalue())



#---------------------------------------------------------------------

#Above, we can see how our Decision Tree Classifier got converted into something that we can read and understand. Now, using the pydotplus, we will convert this into an image. Let's see how would that look like

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('./119/diabetes.png')
Image(graph.create_png())


#In the chart above, we can hardly make out anything, but each of the internal node has a decision rule using which, it splits the data.


#Can we make this decision tree such that we can understand it? Yes we can!


#We can do that by doing some pruning. If we look at the charts above, we can see that our decision tree goes much deeper from our root node. We can limit the max-depth of a Decision Tree Model as per our convenience. Let's work it out again.
#--------------------------------------------------------------------------------

clf = DecisionTreeClassifier(max_depth=3)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Here, we can see that by reducing the maximum depth we want our decision tree to go, we have also achieved a higher accuracy. Let's visualise this again!

#---------------------------------------------------------------------------------
dot_data = StringIO() #Where we will store the data from our decision tree classifier as text.

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


#Here, we can see that the tree is much more readable and understandable. We set the max-depth to 3, so it only goes 3 layers down from the root node.


#This pruned model is less complex, explainable, easy to understand and more accurate than the previous decision tree plot.

#------------------------------------------------------------------------

#Conclusion
#By looking at this chart, we can say with almost 75% accuracy that a person who's

#Glucose is greater than 129.5 and,
#BMI is greater than 27.85
#Is more prone to be a Diabetes Patient.
