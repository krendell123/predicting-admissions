# Testing Python with Weka

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import Classifier, Evaluation
import weka.core.dataset as ds
from weka.core.dataset import Instances
from weka.core.dataset import Instance
import weka.plot.classifiers as plot_cls

jvm.start()

try: 
	# LOAD FILE - load either sixFeature.csv (domain feature selection) or elevenFeatures.csv (no feature selection)
	loader = Loader(classname="weka.core.converters.CSVLoader")
	data = loader.load_file("sixFeatures.csv")

	# PREPROCESS - turn numeric to nominal
	numToNom = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R","first-last"])
	numToNom.inputformat(data)
	data = numToNom.filter(data)
	data.class_is_last()
	print(data.summary(data))

	# CLASSIFIERS
	classifiers = [
		("Bayesian Network", Classifier(classname="weka.classifiers.bayes.BayesNet")),
	#	("Decision Tree", Classifier(classname="weka.classifiers.trees.J48")),
	#	("Logistic Regression", Classifier(classname="weka.classifiers.functions.Logistic")),
	#	("Multilayer Perceptron", Classifier(classname="weka.classifiers.functions.MultilayerPerceptron")),
	#	("Naive Bayes", Classifier(classname="weka.classifiers.bayes.NaiveBayes")),
	#	("Nearest Neighbour", Classifier(classname="weka.classifiers.lazy.IBk"))),
	]

	# EVALUATION
	for name, cls in classifiers:
		print(name)
		evaluation = Evaluation(data)
		evaluation.crossvalidate_model(cls, data, 10, Random(42))
		print(evaluation.summary())
		print(evaluation.class_details())
		print("areaUnderROC/1: " + str(evaluation.area_under_roc(1)))
		print("numFalseNegatives: " + str(evaluation.num_false_negatives(1)))
		print("numTrueNegatives: " + str(evaluation.num_true_negatives(1)))
		print("numFalsePositives: " + str(evaluation.num_false_positives(1)))
		print("numTruePositives: " + str(evaluation.num_true_positives(1)))

	jvm.stop()

except:
	print("runtime error")
	jvm.stop()
