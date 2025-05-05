from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import csv
from time import sleep
from datetime import datetime
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import mediapipe as mp
import json
from keras.models import load_model
from Realtime_SER import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mplcyberpunk
from export_util import *
import librosa
import librosa.display
import multiprocessing
import fitz 
input_frame = [None]
result_frames = [None, None]
final_model_op = {"fer":None, "pose":None, "speech":None}

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

time_sec_list=['0']

# conn = sqlite3.connect('database.db')
# c = conn.cursor()
# c.execute("CREATE TABLE USER(NAME VARCHAR(30) NOT NULL, EMAIL VARCHAR(255));")

class Ui_StartWindow(object):
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1100, 700)
        MainWindow.setStyleSheet("background-color: rgb(31, 31, 31);")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QRect(300, 90, 500, 350))
        self.label.setText("")
        self.label.setPixmap(QPixmap("cosmoclutch_logo.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QRect(400, 450, 300, 50))
        self.lineEdit.setStyleSheet("border: 1px solid white;\n"
"border-radius: 10px;\n"
"font: 14pt \"Noto Sans\";\n"
"background-color: rgb(59, 56, 56);\n"
"color: rgb(255, 255, 255);\n"
"text-align:center;")
        self.lineEdit.setMaxLength(100)
        self.lineEdit.setEchoMode(QLineEdit.Normal)
        self.lineEdit.setCursorPosition(0)
        self.lineEdit.setAlignment(Qt.AlignCenter)
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(475, 590, 150, 50))
        font = QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet("QPushButton {\n"
"color: rgb(255, 255, 255);\n"
"font: 14pt \"Noto Sans\";\n"
"text-align: center;\n"
"border: 1px solid transparent;\n"
"border-radius: 10px;\n"
"    background-color: rgb(181, 99, 195);\n"
"padding: 5px 10px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(153, 81, 166);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: rgb(120, 63, 131);\n"
"}\n"
"")
        self.pushButton.setDefault(True)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.lineEdit_2 = QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QRect(400, 510, 300, 50))
        self.lineEdit_2.setStyleSheet("border: 1px solid white;\n"
"font: 14pt \"Noto Sans\";\n"
"border-radius: 10px;\n"
"background-color: rgb(59, 56, 56);\n"
"color: rgb(255, 255, 255);\n"
"text-align:center;")
        self.lineEdit_2.setMaxLength(200)
        self.lineEdit_2.setCursorPosition(5)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QRect(0, 690, 1100, 10))
        self.progressBar.setStyleSheet("QProgressBar {\n"
"border: 0px solid transparent;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"\n"
"    background-color: rgb(141, 76, 152);\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QRect(5, 670, 500, 16))
        self.label_2.setStyleSheet("color: rgb(204, 204, 204);")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.pushButton.clicked.connect(lambda: self.start_model(MainWindow))

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cosmo Clutch"))#changed
        self.lineEdit.setText(_translate("MainWindow", "Name"))
        self.pushButton.setText(_translate("MainWindow", "Start Model"))
        self.lineEdit_2.setText(_translate("MainWindow", "Email"))
        self.label_2.setText(_translate("MainWindow", ""))

    def start_model(self, Window):
        global main_ui
        self.label_2.setText("Loading Models")
        self.progressBar.setProperty("value", 0)
        with open('details.txt', 'w') as file:
            file.write(self.lineEdit.text()+','+self.lineEdit_2.text())
        Window.setCursor(QCursor(Qt.WaitCursor))
        self.load_all_models()
        Window.hide()
        self.MainWindow = QMainWindow()
        main_ui = Ui_MainWindow()
        main_ui.setupUi(self.MainWindow)
        self.MainWindow.showMaximized()

    def load_all_models(self):
        global mp_pose, pose, face_classifier, classifier, detector1, detector2, detector3, detector4, detector5, detector6
        self.label_2.setText("Loading POSE")
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.progressBar.setProperty("value", 25)
        self.label_2.setText("Loading FER")
        face_classifier = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
        classifier =load_model('Model/Emotion_vgg.h5')
        self.progressBar.setProperty("value", 50)

        estimators = get_best_estimators(True)
        estimators_str, estimator_dict = get_estimators_name(estimators)
        print("Loading estimators: {}".format(estimators_str))

        features = ["mfcc", "chroma", "mel"]
        detector1 = EmotionRecognizer(estimator_dict["SVC"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        detector2 = EmotionRecognizer(estimator_dict["RandomForestClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        detector3 = EmotionRecognizer(estimator_dict["GradientBoostingClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        detector4 = EmotionRecognizer(estimator_dict["KNeighborsClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        detector5 = EmotionRecognizer(estimator_dict["MLPClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        detector6 = EmotionRecognizer(estimator_dict["BaggingClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
        self.label_2.setText("Loading SVC")
        detector1.train()
        print("SVC Ready")
        self.progressBar.setProperty("value", 59)
        self.label_2.setText("Loading RandomForestClassifier")
        detector2.train()
        print("RandomForestClassifier Ready")
        self.progressBar.setProperty("value", 68)
        self.label_2.setText("Loading GradientBoostingClassifier")
        detector3.train()
        print("GradientBoostingClassifier Ready")
        self.progressBar.setProperty("value", 77)
        self.label_2.setText("Loading KNeighborsClassifier")
        detector4.train()
        print("KNeighborsClassifier Ready")
        self.progressBar.setProperty("value", 86)
        self.label_2.setText("Loading MLPClassifier")
        detector5.train()
        print("MLPClassifier Ready")
        self.progressBar.setProperty("value", 95)
        self.label_2.setText("Loading BaggingClassifier")
        detector6.train()
        print("BaggingClassifier Ready")
        self.progressBar.setProperty("value", 100)
        self.label_2.setText("Done")
        time.sleep(3)

        print("Test accuracy score SVC : {:.3f}%".format(detector4.test_score()*100))
        print("Test accuracy score RandomForestClassifier : {:.3f}%".format(detector2.test_score()*100))
        print("Test accuracy score GradientBoostingClassifier : {:.3f}%".format(detector1.test_score()*100))
        print("Test accuracy score KNeighborsClassifier : {:.3f}%".format(detector6.test_score()*100))
        print("Test accuracy score MLPClassifier : {:.3f}%".format(detector3.test_score()*100))
        print("Test accuracy score BaggingClassifier : {:.3f}%".format(detector5.test_score()*100))

class PDFViewer(QMainWindow):
    def __init__(self, filename):
        super().__init__()

        self.setWindowTitle("PDF Viewer")
        self.setGeometry(100, 100, 600, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.pdf_document = None
        self.current_page = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.load_pdf("Report/"+filename)

        self.prev_button = QPushButton("Previous Page", self)
        self.prev_button.clicked.connect(self.prev_page)
        self.layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Page", self)
        self.next_button.clicked.connect(self.next_page)
        self.layout.addWidget(self.next_button)

    def load_pdf(self, pdf_file):
        self.pdf_document = fitz.open(pdf_file)
        self.current_page = 0
        self.show_page(self.current_page)

    def show_page(self, page_number):
        if self.pdf_document is not None and 0 <= page_number < len(self.pdf_document):
            page = self.pdf_document[page_number]
            image = page.get_pixmap()
            qt_image = QImage(image.samples, image.width, image.height, image.stride, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))
            self.setWindowTitle(f"PDF Viewer - Page {page_number + 1}/{len(self.pdf_document)}")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page(self.current_page)

    def next_page(self):
        if self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.show_page(self.current_page)

class Ui_Export(QDialog):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.setWindowTitle("Export")
        self.resize(600, 600)
        self.setStyleSheet("background-color: rgb(31, 31, 31);")
        self.label = QLabel(self)
        self.label.setGeometry(QRect(160, 50, 280, 50))
        self.label.setStyleSheet("font: 26pt \"Segoe UI\";\n"
"color: rgb(255, 255, 255);")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setText("Scan to download")
        self.label_2 = QLabel(self)
        self.label_2.setGeometry(QRect(125, 135, 350, 350))
        self.label_2.setText("")
        # self.label_2.setPixmap(QPixmap("qr.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.pushButton = QPushButton(self)
        self.pushButton.setGeometry(QRect(225, 520, 150, 50))
        font = QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet("QPushButton {\n"
"color: rgb(255, 255, 255);\n"
"font: 14pt \"Noto Sans\";\n"
"text-align: center;\n"
"border: 1px solid transparent;\n"
"border-radius: 10px;\n"
"    background-color: rgb(181, 99, 195);\n"
"padding: 5px 10px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(153, 81, 166);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: rgb(120, 63, 131);\n"
"}\n"
"")
        self.pushButton.setDefault(True)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("View PDF")
        self.pushButton.clicked.connect(self.View_PDF)
        # QMetaObject.connectSlotsByName(self)

    def View_PDF(self):
        self.window = PDFViewer(self.filename)
        self.window.show()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setMinimumSize(QSize(1920, 1080))
        MainWindow.setMaximumSize(QSize(1920, 1080))

        self.filename = None

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QSize(1920, 1057))
        self.centralwidget.setMaximumSize(QSize(1920, 1057))
        self.centralwidget.setStyleSheet("background-color: rgb(31, 31, 31);")
        self.centralwidget.setObjectName("centralwidget")

        self.Topbar = QFrame(self.centralwidget)
        self.Topbar.setGeometry(QRect(0, 0, 1920, 80))
        self.Topbar.setStyleSheet("background-color: rgb(61, 61, 61);\n")
        self.Topbar.setFrameShape(QFrame.StyledPanel)
        self.Topbar.setFrameShadow(QFrame.Raised)
        self.Topbar.setObjectName("Topbar")

        self.TitleLabel = QLabel(self.Topbar)
        self.TitleLabel.setGeometry(QRect(710, 0, 500, 80))
        self.TitleLabel.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "font: 28pt \"Segoe UI\";")
        self.TitleLabel.setObjectName("TitleLabel")
        self.TitleLabel.setText("Cosmo Clutch")
        self.TitleLabel.setAlignment(Qt.AlignCenter)

        self.Speech_label = QLabel(self.Topbar)
        self.Speech_label.setGeometry(QRect(1700, 0, 200, 80))
        self.Speech_label.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "font: 18pt \"Segoe UI\";")
        self.Speech_label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.Speech_label.setObjectName("Speech_label")
        self.Speech_label.setText("Idle")

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setGeometry(QRect(0, 80, 1920, 500))
        self.frame_2.setStyleSheet("background-color: rgb(43, 43, 43);")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        self.FERframe = QFrame(self.frame_2)
        #orig self.FERframe.setGeometry(QRect(5, 10, 640, 480))
        self.FERframe.setGeometry(QRect(1270, 10, 640, 480))
        self.FERframe.setStyleSheet("background-color: rgb(0, 0, 0); border-radius: 10px;")
        self.FERframe.setFrameShape(QFrame.StyledPanel)
        self.FERframe.setFrameShadow(QFrame.Raised)
        self.FERframe.setObjectName("FERframe")

        self.FER_label = QLabel(self.FERframe)
        self.FER_label.setGeometry(QRect(0, 0, 640, 480))
        self.FER_label.setObjectName("FER_label")

        self.POSE_label = QLabel(self.frame_2)
        #orig self.POSE_label.setGeometry(QRect(1270, 10, 640, 480))
        self.POSE_label.setGeometry(QRect(5, 10, 640, 480))
        self.POSE_label.setStyleSheet("background-color: rgb(0, 0, 0); border-radius: 10px;")
        self.POSE_label.setObjectName("POSE_label")

        
        
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(0, 580, 1920, 451))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        plt.style.use("cyberpunk")
        plt.rcParams['figure.facecolor'] = '#1f1f1f'
        plt.rcParams['axes.facecolor'] = '#1f1f1f'
        self.matplotlib_Graphs = plt.figure()
        self.canvas_Graphs = FigureCanvas(self.matplotlib_Graphs)
        self.canvas_Graphs.setParent(self.frame)
        self.canvas_Graphs.setGeometry(QRect(0, 5, 1915, 441))
        self.canvas_Graphs.setStyleSheet("border: 2px solid black; border-radius: 10px; background-color: black;")
        self.g1 =  self.matplotlib_Graphs.add_subplot(121)
        self.g2 =  self.matplotlib_Graphs.add_subplot(122)
        self.g1.set(xlabel='Time', ylabel='Emotion', title='Facial Emotion Recognition & Pose Estimation')
        self.g2.set(xlabel='Time', ylabel='Emotion', title='Speech Emotion Recognition')
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 1920, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionExport = QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionExport.triggered.connect(self.export_data)
        self.actionExport.setEnabled(False)

        self.actionStart = QAction(MainWindow)
        self.actionStart.setObjectName("actionStart")
        self.actionStart.triggered.connect(self.start_model)

        self.actionStop = QAction(MainWindow)
        self.actionStop.setObjectName("actionStop")
        self.actionStop.triggered.connect(self.stop_model)

        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exit_app)

        self.menuFile.addAction(self.actionStart)
        self.menuFile.addAction(self.actionStop)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)

        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)  

        self.Worker2 = Worker2()

        self.timer_graph = QTimer()
        self.timer_graph.setInterval(1000)
        self.timer_graph.timeout.connect(self.animate)

        self.timer_csv = QTimer()
        self.timer_csv.setInterval(1000)
        self.timer_csv.timeout.connect(self.update_files)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExport.setText(_translate("MainWindow", "Export"))
        self.actionStart.setText(_translate("MainWindow", "Start"))
        self.actionStop.setText(_translate("MainWindow", "Stop"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        
      
    def start_model(self):
        self.Worker1.start()
        self.Worker2.start()
        self.timer_graph.start()
        # self.timer_txt.start()
        self.timer_csv.start()
        self.Speech_label.setText("Listening...")
        self.actionExport.setEnabled(False)
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        with open('details.txt', 'r') as file:
            self.details = file.read().split(',')
        self.filename = "data_" + self.details[0] + '_' + date + ".csv"
        self.csv_file = open("CSV_data/"+self.filename, 'a', newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['Time', 'FER', 'POSE', "SER"])

    def stop_model(self):
        self.Worker1.stop()
        self.Worker2.stop()
        self.timer_graph.stop()
        # self.timer_txt.stop()
        self.timer_csv.stop()
        self.Speech_label.setText("Idle")
        self.actionExport.setEnabled(True)
        time_sec_list[0] = '0'
        if self.filename:
            self.csv_file.close()

    def export_data(self):
        generate_graph(self.filename)
        FER_str=""
        POSE_str=""
        SER_str=""
        PHY_str=""
        Summary_str=""
        bpm=0
        systolic=0
        diastolic=0
        with open('CSV_data/'+self.filename, 'r') as file:
            reader = csv.reader(file)
            #exclude header
            reader = list(reader)[1:]
            # calculate avg of bpm , systolic and diastolic
            for row in reader:
                bpm += int(row[4])
                systolic += int(row[5])
                diastolic += int(row[6])

            bpm = bpm/len(reader)
            systolic = systolic/len(reader)
            diastolic = diastolic/len(reader)
        
        if bpm < 40 and systolic < 120:
            print('Depression')
            FER_str = content('FER', 'Depression')
            POSE_str = content('POSE', 'Depression')
            SER_str = content('SER', 'Depression')
            PHY_str = content('PHY', 'Depression')
            Summary_str = content('SUMMARY', 'Depression')
        elif bpm >= 90 and systolic>=120 and diastolic >= 80:
            print('Anxiety')
            FER_str = content('FER', 'Anxiety')
            POSE_str = content('POSE', 'Anxiety')
            SER_str = content('SER', 'Anxiety')
            PHY_str = content('PHY', 'Anxiety')
            Summary_str = content('SUMMARY', 'Anxiety')
        else:
            print('Normal')
            FER_str = content('FER', 'Normal')
            POSE_str = content('POSE', 'Normal')
            SER_str = content('SER', 'Normal')
            PHY_str = content('PHY', 'Normal')
            Summary_str = content('SUMMARY', 'Normal')
        ufile = generate_report(self.details[0], FER_str, POSE_str, SER_str, PHY_str, Summary_str)
        # url = Azure_upload(ufile)
        # QR_gen(url)
        # SendMessage(self.details[1], self.details[0], url)
        self.window = Ui_Export(ufile)
        self.window.show()

    def exit_app(self):
        self.Worker1.stop()
        self.Worker2.stop()
        app.exit()

    def ImageUpdateSlot(self, Image):
        self.POSE_label.setPixmap(QPixmap.fromImage(Image[1]['frame']))
        self.FER_label.setPixmap(QPixmap.fromImage(Image[0]['frame']))
        # self.update_csv()
        # self.update_txt()

    def update_files(self):
        if self.filename:
            # self.columns = ['Time', 'FER', 'POSE', "SER", "BPM", "Systolic", "Diastolic"]
            time_sec = time_sec_list[0]
        
            with open('fer.txt', 'a') as f:
                f.write(time_sec + ',' + str(final_model_op["fer"]) + '\n')
            with open('fer.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) > 10:
                    lines = lines[-10:]
            with open('fer.txt', 'w') as f:
                f.writelines(lines)

            with open('pose.txt', 'a') as f:
                f.write(time_sec + ',' + str(final_model_op["pose"]) + '\n')
            with open('pose.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) > 10:
                    lines = lines[-10:]
            with open('pose.txt', 'w') as f:
                f.writelines(lines)

            with open('speech.txt', 'a') as f:
                f.write(time_sec + ',' + str(final_model_op["speech"]) + '\n')
            with open('speech.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) > 10:
                    lines = lines[-10:]
            with open('speech.txt', 'w') as f:
                f.writelines(lines)

            self.writer.writerow([time_sec, final_model_op["fer"], final_model_op["pose"], final_model_op["speech"]])

            time_sec_list[0] = str(int(time_sec_list[0]) + 1)

    def animate(self):

        emotions = ['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise', 'None']
        self.g1.clear()
        self.g2.clear()
        with open( 'fer.txt', 'r') as graph_data:
            lines = graph_data.read().split('\n')
            x_fer = []
            y_fer = []
            for line in lines :
                if len(line) > 1:
                    x, y = line.split(',')
                    x_fer.append(x)
                    y_fer.append(y)
        with open( 'pose.txt', 'r') as graph_data:
            lines = graph_data.read().split('\n')
            x_pose = []
            y_pose = []
            for line in lines :
                if len(line) > 1:
                    x, y = line.split(',')
                    x_pose.append(x)
                    y_pose.append(y)
        self.g1.plot(x_fer, y_fer, color='yellow', label='FER')
        self.g1.plot(x_pose, y_pose, color='red', label='POSE')
        self.g1.legend()
        self.g1.set_yticks(emotions)
        self.g1.set(xlabel='Time', ylabel='Emotion', title='Facial Emotion Recognition & Pose Estimation')
        with open( 'speech.txt', 'r') as graph_data:
            lines = graph_data.read().split('\n')
            x_speech = []
            y_speech = []
            for line in lines :
                if len(line) > 1:
                    x, y = line.split(',')
                    x_speech.append(x)
                    y_speech.append(y)
        self.g2.plot(x_speech, y_speech, color='red')
        self.g2.set_yticks(emotions)
        self.g2.set(xlabel='Time', ylabel='Emotion', title='Speech Emotion Recognition')
        self.canvas_Graphs.draw()
    def aud_anim(self):
        self.ax1.clear()
        self.ax2.clear()
        file_path = 'test.wav'  
        y, sr = librosa.load(file_path)
        self.ax2 = self.audio_graphs.add_subplot(212)
        self.mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.waveshow(y, sr=sr)
        self.ax2.set_title('Wave plot')
        self.ax1 = self.audio_graphs.add_subplot(311)
        librosa.display.specshow(librosa.power_to_db(self.mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
        self.ax1.set_title('Mel Spectrogram')

    # def update_txt(self):
    #     time_sec = time_sec_list[0]
        
    #     with open('fer.txt', 'a') as f:
    #         f.write(time_sec + ',' + str(final_model_op["fer"]) + '\n')
    #     with open('fer.txt', 'r') as f:
    #         lines = f.readlines()
    #         if len(lines) > 10:
    #             lines = lines[-10:]
    #     with open('fer.txt', 'w') as f:
    #         f.writelines(lines)

    #     with open('pose.txt', 'a') as f:
    #         f.write(time_sec + ',' + str(final_model_op["pose"]) + '\n')
    #     with open('pose.txt', 'r') as f:
    #         lines = f.readlines()
    #         if len(lines) > 10:
    #             lines = lines[-10:]
    #     with open('pose.txt', 'w') as f:
    #         f.writelines(lines)

    #     with open('speech.txt', 'a') as f:
    #         f.write(time_sec + ',' + str(final_model_op["speech"]) + '\n')
    #     with open('speech.txt', 'r') as f:
    #         lines = f.readlines()
    #         if len(lines) > 10:
    #             lines = lines[-10:]
    #     with open('speech.txt', 'w') as f:
    #         f.writelines(lines)

    #     time_sec_list[0] = str(int(time_sec_list[0]) + 1)

def FER():

    frame = input_frame[0].copy()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    label = None
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = classifier.predict(roi)[0]
            if preds.argmax()!=0:
                label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            label="No Face Found"
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    final_model_op["fer"] = label
    result_frames[0] = {'frame' : frame, 'label' : label}

def POSE():
    
    # Define the emotion labels
    emotion_labels = {
        "Happy": 0,
        "Sad": 0,
        "Angry": 0,
        "Neutral": 0
    }

    # Define the thresholds for emotion detection
    happy_threshold = 160
    sad_threshold = 120
    angry_threshold = 30

    # Function to detect emotion based on body posture
    def detect_emotion(landmarks):
        # Get the coordinates of relevant landmarks
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate the angle between shoulders and wrists
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Detect emotions based on the angle
        if left_angle > happy_threshold and right_angle > happy_threshold:
            emotion_labels["Happy"] += 1
        elif left_angle < sad_threshold and right_angle < sad_threshold:
            emotion_labels["Sad"] += 1
        elif left_angle < angry_threshold or right_angle < angry_threshold:
            emotion_labels["Angry"] += 1
        else:
            emotion_labels["Neutral"] += 1

    # Function to calculate angle between three landmarks
    def calculate_angle(a, b, c):
        vector1 = [a.x - b.x, a.y - b.y]
        vector2 = [c.x - b.x, c.y - b.y]
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
        magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    frame=input_frame[0].copy()
    # frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set flag to enable drawing landmarks on the image
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    # Detect pose landmarks
    results = pose.process(image)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Get emotion from body posture
        detect_emotion(results.pose_landmarks)

    # Draw pose landmarks on the image
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)

    # Convert the RGB image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    final_model_op["pose"] = max(zip(emotion_labels.values(), emotion_labels.keys()))[1]
    result_frames[1] = {'frame' : image, 'label' : max(zip(emotion_labels.values(), emotion_labels.keys()))[1]}
        
class Worker1(QThread):
    ImageUpdate = pyqtSignal(list)
    def __init__(self):
        super(Worker1, self).__init__()
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                input_frame[0] = frame
                FER()
                POSE()

                fer_op = cv2.cvtColor(result_frames[0]['frame'], cv2.COLOR_BGR2RGB)
                fer_qt = QImage(fer_op.data, fer_op.shape[1], fer_op.shape[0], QImage.Format_RGB888)
                pose_op = cv2.cvtColor(result_frames[1]['frame'], cv2.COLOR_BGR2RGB)
                pose_qt = QImage(pose_op.data, pose_op.shape[1], pose_op.shape[0], QImage.Format_RGB888)

                fer_Pic = fer_qt.scaled(640, 480, Qt.KeepAspectRatio)
                pose_Pic = pose_qt.scaled(640, 480, Qt.KeepAspectRatio)

                op_list = [{'label' : result_frames[0]['label'], 'frame' : fer_Pic}, {'label' : result_frames[1]['label'], 'frame' : pose_Pic}]
                self.ImageUpdate.emit(op_list)
    def stop(self):
        self.ThreadActive = False
        self.quit()
        self.wait()

class Worker2(QThread):
    global main_ui
    EmotionUpdate = pyqtSignal(str)
    def __init__(self):
        super(Worker2, self).__init__()
    def run(self):
        self.ThreadActive = True
        print("Please talk")
        
        while self.ThreadActive:
            record_to_file("test.wav")
            # main_ui.aud_anim()
            results = [detector1.predict("test.wav"), detector2.predict("test.wav"), detector3.predict("test.wav"), detector4.predict("test.wav"), detector5.predict("test.wav"), detector6.predict("test.wav")]
            result = find_final_emotion(results)
            result = result.capitalize()
            print(result, results)
            final_model_op["speech"] = result
            self.EmotionUpdate.emit(result)

    def stop(self):
        self.ThreadActive = False
        self.quit()
        self.wait()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    StartWindow = QMainWindow()
    ui = Ui_StartWindow()
    ui.setupUi(StartWindow)
    StartWindow.show()
    # Window.showMaximized()
    
    sys.exit(app.exec_())
















