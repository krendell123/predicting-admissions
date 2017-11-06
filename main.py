from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import admission
import model
import inputData

class AdmissionApp(QtWidgets.QMainWindow, admission.Ui_MainWindow):

	triageList = inputData.triage
	booleanList = inputData.boolean
	ageList = inputData.age
	hourList = inputData.hour
	genderList = inputData.gender
	presentingList = inputData.presenting
	model = model.AdmitModel(modelType="NN") # Model type is changable

	def __init__(self, parent=None):
		super(AdmissionApp, self).__init__(parent)
		self.setupUi(self)
		self.triageLabel.setToolTip("1 - immediantly life-threatening, 2 - imminently life-threatening, 3 - potentially life-threatening, 4 - potentially serious, 5 - less urgent")
		self.ambulanceLabel.setToolTip("The patient arrived at ED by ambulance")
		self.prevAdmitLabel.setToolTip("The patient has been admitted to hospital in the last 30 days")
		self.presentingLabel.setToolTip("Select a presenting problem category")
		self.triageBox.addItems(sorted(self.triageList.keys()))
		self.ambulanceBox.addItems(self.booleanList.keys())
		self.prevAdmitBox.addItems(self.booleanList.keys())
		self.ageBox.addItems(sorted(self.ageList.keys()))
		self.presentingBox.addItems(sorted(self.presentingList.keys()))
		self.presentingBox.currentIndexChanged.connect(self.update_presenting)
		self.calcBtn.clicked.connect(self.calculate_admission)

	# Updates presenting problem list widget with all subcategories of choosen presenting problem category
	def update_presenting(self):
		self.listWidget.clear()
		pres_cat = self.presentingBox.currentText()
		pres_index = inputData.presenting[pres_cat]
		pres_dict = inputData.presenting_list[pres_index - 1]
		self.listWidget.addItems(sorted(pres_dict.keys()))

	# Gets user inputs from widget and predicts admission
	def calculate_admission(self):

		# Get input values for each feature
		tr_value = self.triageList[self.triageBox.currentText()]
		amb_value = self.booleanList[self.ambulanceBox.currentText()]
		age_value = self.ageList[self.ageBox.currentText()]
		prevAdmit_value = self.booleanList[self.prevAdmitBox.currentText()]

		# Get value for presenting problem
		pres_cat = self.presentingBox.currentText()
		pres_index = inputData.presenting[pres_cat]

		# If subcategory not selected then we just use the category value
		if self.listWidget.currentItem() is None:
			pres_value = str(pres_index)
		else:
			pres_dict = inputData.presenting_list[pres_index - 1]
			pres_value = pres_dict[self.listWidget.currentItem().text()]

		# Calculate admission
		admit = self.model.calculate_admission(tr_value, amb_value, age_value, prevAdmit_value, pres_value)

		# Display result in QMessage Box
		if admit[0] == "1":
			msg = QtWidgets.QMessageBox.information(self, "Admission Status", "Based on the inputted triage notes the patient is likely to be ADMITTED")
		else:
			msg = QtWidgets.QMessageBox.information(self, "Admission Status", "Based on the inputted triage notes the patient is likely to be DISCHARGED")


def main():
	app = QtWidgets.QApplication(sys.argv)
	form = AdmissionApp()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
