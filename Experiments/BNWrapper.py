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
	# LOAD FILE
	loader = Loader(classname="weka.core.converters.CSVLoader")
	data = loader.load_file("/project/RDS-FEI-HospitalAdmissions-RW/ArtemisTraining/20Presenting/1.5Wrapper/elevenFeatures.csv")

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
	
	for name, cls in classifiers:
		# Use CFS to find subset of attributes
		print("Attribute Selection " + name)
		search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "6"])
		evaluation = ASEvaluation(classname="weka.attributeSelection.WrapperSubsetEval", options=["-B", "weka.classifiers.bayes.BayesNet", "-F", "5", "-T", "0.01", "-R", "1", "--"])
		attsel = AttributeSelection()
		attsel.search(search)
		attsel.evaluator(evaluation)
		attsel.select_attributes(data)
		print("# attributes: " + str(attsel.number_attributes_selected))
		print("attributes: " + str(attsel.selected_attributes))
		print("result string:\n" + attsel.results_string)

		print(name + " after attribute selection")

		# Find indicies of attributes to delete
		attributes_to_delete = []
		print(data.num_attributes)
		print(data.class_index)
		print(attsel.selected_attributes)
		for x in range(0, data.num_attributes):
						if x not in attsel.selected_attributes and x != data.class_index:
								attributes_to_delete.append(x)
		print(attributes_to_delete)

		# Create copy of data and delete unneccassary attributes
		dataSubset = data.copy_instances(data)
		for x in range(dataSubset.num_attributes-1, -1, -1):
				if x in attributes_to_delete:
						dataSubset.delete_attribute(x)
		print(dataSubset.summary(dataSubset))
		print(data.class_index)

		# Repeat classification on data subset
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
