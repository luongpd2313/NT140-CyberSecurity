import pandas 
import numpy 
import matplotlib.pyplot as pyplot
import matplotlib.pylab as pylab
from sklearn.feature_extraction.text import CountVectorizer
import math
from collections import Counter
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

import lime
import lime.lime_tabular

import sklearn
import shap
from sklearn.model_selection import train_test_split

import numpy as np
from alibi.explainers import AnchorTabular

import requests

from OTXv2 import OTXv2
from OTXv2 import IndicatorTypes
import json
import pprint


def plot_cm(cm, labels):
    percent = (cm*100.0)/numpy.array(numpy.matrix(cm.sum(axis=1)).T)
    print( 'Confusion Matrix Stats' )
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print( "%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()) )
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm')
    pylab.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pylab.xlabel('Predicted')
    pylab.ylabel('True')
    pylab.show()




print('#1 Read The Data: dataset/data_exported_7features.csv')

#Num,No,Domainname,Label,Entropy,REAlexa,REConficker,RECryptolocker,REGoz,REMatsnu,RENew_goz,REPushdo,RERamdo,RERovnix,RETinba,REZeus,MinREBotnets,InformationRadius,ClassificationResult,Result,CharLength,LabelBinary,TreeNewFeature,nGramReputation_Alexa
data = pandas.read_csv('data_exported_7features.csv', header=0, encoding='utf-8')

print(data.head())
# print(data[['Entropy']].min())
# print(data[['REAlexa']].min())
# print(data[['MinREBotnets']].min())
# print(data[['InformationRadius']].min())
# print(data[['Entropy', 'REAlexa', 'MinREBotnets', 'InformationRadius']].describe())
# print(data[['CharLength']].min())
# print(data[['TreeNewFeature']].min())
# print(data[['nGramReputation_Alexa']].min())
# print(data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa' ]].describe())


print('#2 Calculate / Prepare The Data (Features Selection)')

print('#2 Calculate / Prepare The Data (Features Selection)')

X = data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']].to_numpy()
# print(X[:5])
# print(numpy.where(numpy.isnan(X)))
# print(X.dtype)
# print(X[:5])
y = numpy.array(data['Label'].tolist())
# print(y[:5])
yBinary = numpy.array(data['LabelBinary'].tolist())
# print(yBinary[:5])

print(data['Domainname'])
print(X)
print(y)
print(yBinary)


# 1. Univariate Selection (chi-squared (chi²) statistical test)
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,yBinary)
# dfscores = pandas.DataFrame(fit.scores_)
# dfcolumns = pandas.DataFrame(xColumns)
# featureScores = pandas.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']
# print(featureScores.nlargest(10,'Score'))
# featureScores.nlargest(10,'Score').plot(x='Specs', y='Score', kind='barh')
# pyplot.title('Univariate Selection (chi-squared (chi²) statistical test)')
# pyplot.savefig("Chi-Squared.png", dpi=300, papertype='a4')

# 2. Feature Importance (from Tree Based Classifiers)
# model = ExtraTreesClassifier()
# model.fit(X,yBinary)
# print(model.feature_importances_)
# feat_importances = pandas.Series(model.feature_importances_, index=xColumns)
# feat_importances.nlargest(10).plot(kind='barh')
# pyplot.title('Feature Importance (from Tree Based Classifiers)')
# pyplot.savefig("FeatureImportance.png", dpi=300, papertype='a4')
# print(feat_importances)

# 3.Correlation Matrix with Heatmap
# headtmapData = data.drop(columns=['Num','No','Domainname','Label','ClassificationResult','Result'])
# corrmat = headtmapData.corr()
# top_corr_features = corrmat.index
# pyplot.figure(figsize=(20,20))
# g=seaborn.heatmap(headtmapData[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# pyplot.title('Correlation Matrix with Heatmap')
# pyplot.savefig("CorrelationMatrixHeatmap.png", dpi=300, papertype='a4')


print('#3 Prepare the trainin testing data')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, test_size=25)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


print('#4 LogisticRegression')

clf1 = LogisticRegression(random_state=1)
clf1.fit(X_train, y_train)


print('#5 RandomForestClassifier')

clf2 = RandomForestClassifier(bootstrap=True, max_depth=None, min_samples_leaf=1,
                              n_estimators=1500, n_jobs=40, oob_score=False,
                              random_state=1, verbose=1)
clf2.fit(X_train, y_train)

print('#6 GaussianNB')

clf3 = GaussianNB()
clf3.fit(X_train, y_train)

print('#7 ExtraTreesClassifier')

clf4 = ExtraTreesClassifier()
clf4.fit(X_train, y_train)

print('#8 VotingClassifier')

clf5 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('etr', clf4)], voting='soft')
clf5.fit(X_train, y_train)

# openModelFilename = "et_botnet.onnx"

# print("\n\nConvert into ONNX format")
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# initial_type = [('float_input', FloatTensorType([None, 4]))]
# print("\ninitial_type")
# print(initial_type)
# onx = convert_sklearn(clf4, initial_types=initial_type)
# print("\nonx")
# print(onx)
# with open(openModelFilename, "wb") as f:
#     f.write(onx.SerializeToString())
# print("\nDONE")

# print("\n\nCompute the prediction with ONNX Runtime")
# import onnxruntime as rt
# import numpy
# sess = rt.InferenceSession(openModelFilename)
# print("\nsess")
# print(sess)
# input_name = sess.get_inputs()[0].name
# print("\ninput_name")
# print(input_name)
# label_name = sess.get_outputs()[0].name
# print("\nlabel_name")
# print(label_name)
# pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
# print("\npred_onx")
# print(pred_onx)
# print("\nDONE")



print("#11 Comparison")

for clf, label in zip([clf1, clf2, clf3, clf4, clf5], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                       'Extra Tree', 'Ensemble']):
    scores = model_selection.cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
print(X_test)
print(y_test)

y_pred = clf1.predict(X_test)
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plot_cm(cm, labels)

y_pred = clf2.predict(X_test)
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plot_cm(cm, labels)

y_pred = clf3.predict(X_test)
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plot_cm(cm, labels)

y_pred = clf4.predict(X_test)
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plot_cm(cm, labels)

y_pred = clf5.predict(X_test)
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plot_cm(cm, labels)

print('#12 LIME, model-agnostic, local explainer')

feature_names = ['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']
class_names = ['dga', 'legit']
limeexplainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
i = numpy.random.randint(0, X_test.shape[0])

print("DOMAIN = " + data['Domainname'][i])

mondai = data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']].to_numpy()[i]
print("CharLength, TreeNewFeature, nGramReputation_Alexa, MinREBotnets")
print(mondai)

kotai = data['Label'][i] 
print("Ground Truth ANSWER = " + kotai)

coba = []
coba.append(mondai)
y_pred = clf2.predict(coba)
print("PREDICTION = " + str(y_pred[0]) )

if(kotai == str(y_pred[0])):
    print("CORRECT Prediction :)")
else:
    print("WRONG Prediction :(")

exp = limeexplainer.explain_instance(mondai, clf2.predict_proba)
exp.show_in_notebook()
print(exp.as_list())
print(exp.as_map())

i = 1400000

print("DOMAIN = " + data['Domainname'][i])

mondai = data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']].to_numpy()[i]
print("CharLength, TreeNewFeature, nGramReputation_Alexa, MinREBotnets")
print(mondai)

kotai = data['Label'][i] 
print("Ground Truth ANSWER = " + kotai)

coba = []
coba.append(mondai)
y_pred = clf2.predict(coba)
print("PREDICTION = " + str(y_pred[0]) )

if(kotai == str(y_pred[0])):
    print("CORRECT Prediction :)")
else:
    print("WRONG Prediction :(")

exp = limeexplainer.explain_instance(mondai, clf2.predict_proba)
exp.show_in_notebook()
print(exp.as_list())
print(exp.as_map())

print('#13 SHAPE, Model agnostic with KernelExplainer')
# print the JS visualization code to the notebook
shap.initjs()

columns = ['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']

X_traindf = pandas.DataFrame(X_train, columns=columns)
print(X_traindf)

X_testdf = pandas.DataFrame(X_test, columns=columns)
print(X_test)

print(y_train)
print(y_test)



print(clf2)

shapexplainer = shap.KernelExplainer(clf2.predict_proba, X_traindf, link="logit")
shap_values = shapexplainer.shap_values(X_testdf, nsamples=100)

i = 0
while i < len(shap_values):
    j = 0
    while j < len(shap_values[i]):
        k = 0
        while k < len(shap_values[i][j]):
            if(not numpy.isfinite(shap_values[i][j][k])):
                shap_values[i][j][k] = 0
            k += 1
        j += 1
    i += 1
    
shap.force_plot(shapexplainer.expected_value[0], shap_values[0][0,:], X_testdf.iloc[0,:], link="logit")
if not numpy.all(numpy.isfinite( shapexplainer.expected_value )):
    print("ERROR A")
if not numpy.all(numpy.isfinite( shap_values[0] )):
    print("ERROR B1")
if not numpy.all(numpy.isfinite( shap_values )):
    print("ERROR B2")
if not numpy.all(numpy.isfinite( X_testdf )):
    print("ERROR C")

    
print(shapexplainer.expected_value[0])
print(len(shap_values[0]))
print(len(X_testdf))

shap.force_plot(shapexplainer.expected_value[0], shap_values[0], X_testdf, link="logit")
shap.summary_plot(shap_values, X_testdf)
shap.summary_plot(shap_values, X_testdf, plot_type="bar")
shap.dependence_plot(0, shap_values[0], X_testdf)
shap.dependence_plot(1, shap_values[0], X_testdf)
shap.dependence_plot(2, shap_values[0], X_testdf)
shap.dependence_plot(3, shap_values[0], X_testdf)



print('#14 What If & Counterfactual Explanations')
columnswif = ['Domainname', 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets', 'Label', 'LabelBinary']
Xwif = data[columnswif]
ywif = data['LabelBinary']

X_trainwif, X_testwif, y_trainwif, y_testwif = train_test_split(Xwif, ywif, train_size=100, test_size=25)
print(X_testwif)
print(y_testwif)


# Return model predictions and SHAP values for each inference.
def custom_predict_with_shap(examples_to_infer):
    #print(examples_to_infer)

    # Delete columns not used by model
    #model_inputs = examples_to_infer[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']]
    model_inputs = numpy.delete(
        numpy.array(examples_to_infer), [0, 5, 6], axis=1).tolist()
    #print(model_inputs)
    
    # Get the class predictions from the model.
    results = clf2.predict(model_inputs)
    #print(results)
    
    preds = []
    for x in results:
        if x =='dga':
            preds.append([0, 1])
        else:
            preds.append([1, 0])
    #print(preds)
    #preds = [[1 - pred[0], pred[0]] for pred in preds]
    
    # Get the SHAP values from the explainer and create a map
    # of feature name to SHAP value for each example passed to
    # the model.
    #shap_output = shapexplainer.shap_values(numpy.array(model_inputs))[0]
    X_test = X_testwif[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']]
    #print(X_test)
    shap_output = shapexplainer.shap_values(X_test)[0]
    
    i = 0
    while i < len(shap_output):
        j = 0
        while j < len(shap_output[i]):
            if(not numpy.isfinite(shap_output[i][j])):
                shap_output[i][j] = 0
            j += 1
        i += 1
    #print(shap_output)
    
    attributions = []
    for shap in shap_output:
        attrs = {}
        for i, col in enumerate(X_test.columns):
            attrs[col] = shap[i]
        attributions.append(attrs)
    ret = {'predictions': preds, 'attributions': attributions}
    #print(ret)
    
    return ret
    
  #@title Invoke What-If Tool for test data and the trained model {display-mode: "form"}

num_datapoints = 2000  #@param {type: "number"}
tool_height_in_px = 1000  #@param {type: "number"}

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

#custom_predict_with_shap(numpy.array(X_testwif).tolist())

#print(X_test)
# Setup the tool with the test examples and the trained classifier
#print(numpy.array(X_testwif).tolist())
config_builder=WitConfigBuilder(numpy.array(X_testwif).tolist(),feature_names=columnswif).set_custom_predict_fn(custom_predict_with_shap).set_target_feature('LabelBinary')


WitWidget(config_builder, height=tool_height_in_px)





print('#14 Anchor explanations')
anchorexplainer = AnchorTabular(clf2.predict_proba, feature_names)

#print(type(np.array(X_trainwif)))
print(np.array(X_trainwif[feature_names]))

anchorexplainer.fit(np.array(X_trainwif[feature_names]), disc_perc=(25, 50, 75))


# missclassification = 7
idx = 10
print( np.array(X_testwif['Domainname'])[idx] )
print( "Ground Truth = " + np.array(X_testwif['Label'])[idx] )

mondai2 = np.array(X_testwif[feature_names])[idx]
print(mondai2)

print('Prediction: ', class_names[anchorexplainer.predictor(mondai2.reshape(1, -1))[0]])



print('#15 Result is explained by: ANCHOR')

# We set the precision threshold to 0.95. 
# This means that predictions on observations where the anchor holds will be the same as the prediction on the explained instance at least 95% of the time.

explanation = anchorexplainer.explain(mondai2, threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)


print('#16 Result is explained by: LIME')

exp = limeexplainer.explain_instance(mondai2, clf2.predict_proba)
exp.show_in_notebook()
print(exp.as_list())
print(exp.as_map())


print('#17 RULE is explained by: SHAP')

shap_values = shapexplainer.shap_values(mondai2)
print(shap_values[0])
print(shapexplainer.expected_value[0])

#print(pandas.DataFrame(mondai2.reshape(1, -1)))

shap.force_plot(shapexplainer.expected_value[0], shap_values[0], pandas.DataFrame(mondai2.reshape(1, -1), columns=feature_names), link="logit")


print('#18 Check by Google Safebrowsing API')
# try this domain = http://nicetelecom.us

url = 'https://safebrowsing.googleapis.com/v4/threatMatches:find?key=AIzaSyC-uuDw56kFYnM1OhXRsRQxnIKUPeOLDdQ'

checkdomain = 'http://nicetelecom.us'
print(checkdomain)

myobj = {
    'client': {
        'clientId': 'coba', 
        'clientVersion': '0.0.1'
    }, 
    'threatInfo': {
        'threatTypes': ['THREAT_TYPE_UNSPECIFIED', 'MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION'], 
        'platformTypes': ['ALL_PLATFORMS'], 
        'threatEntryTypes': ['URL'], 
        'threatEntries': [
            {'url': checkdomain}
        ]
    }
}
print(myobj)

x = requests.post(url, json = myobj)

print(x)

if len(x.text) <= 3:
    print("No information about this domainname on Google Safe Browser database")
else:
    print(x.text)
    
print('#19 Check by OTX AlienVault API')

otx = OTXv2("2097188e3e163989f0e3fba2672f075aa4042fab610c6a27f63f1aaa23c3c3f2")
pp = pprint.PrettyPrinter(indent=4)
# Example domains: 'rghost.net', 'spywaresite.info'

checkdomain = 'rghost.net'
print(checkdomain)

json_data = otx.get_indicator_details_full(IndicatorTypes.DOMAIN, checkdomain)

if len(json_data['malware']['data']) <= 3:
    print("No information about this domainname on OTX AlienVault API database")
else:
    pp.pprint(json_data['malware']['data'])
