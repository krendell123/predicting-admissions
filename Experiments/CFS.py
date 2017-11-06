# Testing Python with Weka

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import Classifier, Evaluation
import weka.core.dataset as ds
from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

jvm.start()

try: 
	# LOAD FILE - use elevenFeature.csv
	loader = Loader(classname="weka.core.converters.CSVLoader")
	data = loader.load_file("/project/RDS-FEI-HospitalAdmissions-RW/ArtemisTraining/20Presenting/1.4CFS/elevenFeatures.csv")

	# PREPROCESS - turn numeric to nominal
	numToNom = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R","first-last"])
	numToNom.inputformat(data)
	data = numToNom.filter(data)
	data.class_is_last()
	print(data.summary(data))
	
	# Use CFS to find subset of attributes
	print("CfsSubsetEval Attribute Selection")
	search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "11"])
	evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
	attsel = AttributeSelection()
	attsel.search(search)
	attsel.evaluator(evaluation)
	attsel.select_attributes(data)
	print("# attributes: " + str(attsel.number_attributes_selected))
	print("attributes: " + str(attsel.selected_attributes))
	print("result string:\n" + attsel.results_string)
	
	# Find indicies of attributes to delete
	attributes_to_delete = []
	for x in range(0, data.num_attributes):
			if x not in attsel.selected_attributes and x != data.class_index:
				attributes_to_delete.append(x)

	# Create copy of data and delete unneccassary attributes
	dataSubset = data.copy_instances(data)
	for x in range(dataSubset.num_attributes-1, -1, -1):
		if x in attributes_to_delete:
			dataSubset.delete_attribute(x)
	print(dataSubset.summary(dataSubset))

	# CLASSIFIERS
	classifiers = [
		("Bayesian Network", Classifier(classname="weka.classifiers.bayes.BayesNet")),
	#	("Decision Tree", Classifier(classname="weka.classifiers.trees.J48")),
	#	("Logistic Regression", Classifier(classname="weka.classifiers.functions.Logistic")),
	#	("Multilayer Perceptron", Classifier(classname="weka.classifiers.functions.MultilayerPerceptron")),
	#	("Naive Bayes", Classifier(classname="weka.classifiers.bayes.NaiveBayes")),
	#	("Nearest Neighbour", Classifier(classname="weka.classifiers.lazy.IBk"))),
	]


	# ATTRIBUTE SELECTION
	for name, cls in classifiers:
		
		print(name + " after IG attribute selection")
		evaluation = Evaluation(dataSubset)
		evaluation.crossvalidate_model(cls, dataSubset, 10, Random(42))
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
