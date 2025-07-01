import sys
import time
import os
import json
import subprocess
import logging
from plyer import notification

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import vstack, hstack

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QDialog, QGraphicsOpacityEffect,
    QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QTextEdit
)
from PySide6.QtGui import QPixmap, QFontDatabase, QFont, QIcon
from PySide6.QtCore import Qt, QThread, Signal, Slot, QPropertyAnimation

BASE_DIR = os.path.expanduser("~/komainud")
LOG_FILE = os.path.join(BASE_DIR, "journal_dump.json")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
PID_FILE = os.path.join(BASE_DIR, "komainu.pid")
MONITOR_PID_FILE = os.path.join(BASE_DIR, "komainu_monitor.pid")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 10000
N_CLUSTERS = 100
CONTAMINATION = 0.0001

def setup_logging():
    os.makedirs(BASE_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(BASE_DIR, "komainu.log"),
                        level=logging.INFO,
                        format=LOG_FORMAT)

def send_notification(title, message):
    try:
        notification.notify(title=title, message=message, app_name='Komainu', timeout=5)
    except Exception as e:
        logging.warning(f"Failed to send notification: {e}")

def dump_journal_logs():
    logging.info("Dumping journal logs (streamed)...")
    send_notification("Komainu", "dumping logs...")
    cmd = ['journalctl', '-o', 'json', '--no-pager']
    start = time.time()
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1) as proc, \
             open(LOG_FILE, 'w', encoding='utf-8') as f:
            count = 0
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    json.dump(obj, f)
                    f.write('\n')
                    count += 1
                except json.JSONDecodeError:
                    logging.warning("Failed to decode a journal line")
                    continue
        logging.info(f"Successfully dumped {count} journal entries to {LOG_FILE}")
    except Exception as e:
        logging.error(f"Failed to stream journal logs: {e}")
        return 'dump unsuccessful'
    end = time.time()
    logging.info(f"Journal dump completed in {end - start:.2f} seconds.")
    send_notification("Komainu", "Initial log dump completed")
    return 'dump successful'

vectorizer = HashingVectorizer(n_features=2**12, alternate_sign=False, norm='l2', stop_words='english')

def extract_features_for_batch(logs_batch):
    timestamps, priorities, messages = [], [], []
    for log in logs_batch:
        ts_str = log.get('__REALTIME_TIMESTAMP')
        try:
            ts_sec = int(ts_str) / 1_000_000_000
        except Exception:
            ts_sec = 0
        timestamps.append(ts_sec)
        msg = log.get('MESSAGE', '')
        if not isinstance(msg, str):
            msg = str(msg)
        messages.append(msg)
        prio_str = log.get('PRIORITY', '5')
        try:
            prio = int(prio_str)
        except:
            prio = 5
        priorities.append(prio)
    start = timestamps[0] if timestamps else 0
    timestamps_rel = [t - start for t in timestamps]
    return np.array(timestamps_rel).reshape(-1, 1), np.array(priorities).reshape(-1, 1), messages

def process_batch_for_training(batch, vectorizer):
    timestamps, priorities, raw_messages = extract_features_for_batch(batch)
    messages = [str(m) if not isinstance(m, str) else m for m in raw_messages]
    ts_arr = np.array(timestamps).reshape(-1, 1)
    prio_arr = np.array(priorities).reshape(-1, 1)
    X_msg = vectorizer.transform(messages)
    X_batch = hstack([ts_arr, prio_arr, X_msg])
    return X_batch, X_msg

def train_models():
    print("Starting training...")
    send_notification("Komainu", "Training models...")
    all_X_batches = []
    all_msg_batches = []
    total_processed_for_training = 0
    print("\n--- Phase: Collecting features for model training ---")
    current_batch_raw_logs = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line)
                current_batch_raw_logs.append(log)
            except json.JSONDecodeError:
                continue
            if len(current_batch_raw_logs) >= BATCH_SIZE:
                X_batch, X_msg = process_batch_for_training(current_batch_raw_logs, vectorizer)
                all_X_batches.append(X_batch)
                all_msg_batches.append(X_msg)
                total_processed_for_training += len(current_batch_raw_logs)
                current_batch_raw_logs = []
        if current_batch_raw_logs:
            X_batch, X_msg = process_batch_for_training(current_batch_raw_logs, vectorizer)
            all_X_batches.append(X_batch)
            all_msg_batches.append(X_msg)
            total_processed_for_training += len(current_batch_raw_logs)
    print(f"Finished collecting features from {total_processed_for_training} log entries.")
    if total_processed_for_training == 0:
        return "No logs to train!"
    X_train = vstack(all_X_batches)
    X_msg_train = vstack(all_msg_batches)
    isolation_model = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42)
    isolation_model.fit(X_train)
    cluster_model = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=1000, n_init='auto')
    cluster_model.fit(X_msg_train)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'hashing_vectorizer.joblib'))
    joblib.dump(isolation_model, os.path.join(MODEL_DIR, 'isolation_forest.joblib'))
    joblib.dump(cluster_model, os.path.join(MODEL_DIR, 'kmeans.joblib'))
    send_notification("Komainu", "Model training completed")
    logging.info("Model training completed successfully.")
    return 'Model trained!'

class WorkerThread(QThread):
    finished = Signal(str)
    def __init__(self, func):
        super().__init__()
        self.func = func
    def run(self):
        result = self.func()
        self.finished.emit(result)

class LoaderDialog(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc;")
        self.setFixedSize(300, 100)
        label = QLabel(message)
        label.setFont(QFont("Cantarell", 12))
        label.setStyleSheet("color: #222222;")
        label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class AnomalyViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Anomaly Reports")
        self.resize(1100, 650)
        self.setModal(True)
        self.setStyleSheet("background-color: #fafafa;")

        layout = QVBoxLayout(self)

        header = QLabel("📌 Detected Anomalies")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #222;")
        layout.addWidget(header)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedSize(100, 30)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #d0ebe2;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #b4dbce;
            }
        """)
        refresh_btn.clicked.connect(self.populate_table)

        layout.addWidget(refresh_btn, alignment=Qt.AlignRight)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Priority", "Cluster", "Message (click row to view)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setStyleSheet("font-size: 13px;")
        self.table.cellDoubleClicked.connect(self.show_full_details)

        layout.addWidget(self.table)
        self.populate_table()

    def populate_table(self):
        log_path = os.path.join(BASE_DIR, "logs/komainu_monitor.log")
        if not os.path.exists(log_path):
            return
        self.entries = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "ANOMALY: " in line:
                    try:
                        data = json.loads(line.split("ANOMALY: ", 1)[1])
                        self.entries.append(data)
                    except Exception:
                        continue
        self.table.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            self.table.setItem(row, 0, QTableWidgetItem(str(entry.get("timestamp", ""))))
            self.table.setItem(row, 1, QTableWidgetItem(str(entry.get("priority", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(entry.get("cluster", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(str(entry.get("message", ""))[:150] + "..."))

    def show_full_details(self, row, _column):
        entry = self.entries[row]
        dlg = QDialog(self)
        dlg.setWindowTitle("📋 Anomaly Detail")
        dlg.resize(800, 500)
        dlg.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(dlg)

        meta = f"""
<b>Timestamp:</b> {entry.get("timestamp", "")}<br>
<b>Priority:</b> {entry.get("priority", "")}<br>
<b>Cluster:</b> {entry.get("cluster", "")}<br><br>
"""
        meta_label = QLabel(meta)
        meta_label.setTextFormat(Qt.RichText)
        layout.addWidget(meta_label)

        msg_box = QTextEdit()
        msg_box.setPlainText(entry.get("message", ""))
        msg_box.setReadOnly(True)
        msg_box.setStyleSheet("background-color: #f2f2f2; font-family: monospace; font-size: 12px;")
        layout.addWidget(QLabel("📨 Message:"))
        layout.addWidget(msg_box)

        summary_box = QTextEdit()
        summary_box.setPlainText(entry.get("summary", ""))
        summary_box.setReadOnly(True)
        summary_box.setStyleSheet("background-color: #f9f9f9; font-family: monospace; font-size: 12px;")
        layout.addWidget(QLabel("📄 Summary:"))
        layout.addWidget(summary_box)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet("background-color: #ddd; font-weight: bold;")
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dlg.exec()

class KomainuApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Komainu")
        self.setFixedSize(800, 500)
        self.setStyleSheet("background-color:rgb(197, 200, 182);")
        QFontDatabase.addApplicationFont("/usr/share/fonts/truetype/cantarell/Cantarell-Regular.ttf")
        self.font_family = "Cantarell"
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        pixmap = QPixmap("komainu.jpg").scaledToHeight(500, Qt.SmoothTransformation)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        title = QLabel("KOMAINU")
        title.setFont(QFont(self.font_family, 28, QFont.Bold))
        title.setStyleSheet("color: #222222;")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("log analyzer")
        subtitle.setFont(QFont(self.font_family, 14))
        subtitle.setStyleSheet("color: #444444;")
        subtitle.setAlignment(Qt.AlignCenter)
        self.dump_button = QPushButton("dump logs")
        self.train_button = QPushButton("train models")
        self.live_button = QPushButton("live monitor")
        self.view_button = QPushButton("view anomalies")
        for btn in [self.dump_button, self.train_button, self.live_button, self.view_button]:
            btn.setFixedSize(220, 40)
            btn.setFont(QFont(self.font_family, 12))
            btn.setCursor(Qt.PointingHandCursor)
        self.dump_button.clicked.connect(lambda: self.run_task(dump_journal_logs, "Dumping logs..."))
        self.train_button.clicked.connect(lambda: self.run_task(train_models, "Training models..."))
        self.live_button.clicked.connect(self.toggle_live_monitor)
        self.view_button.clicked.connect(self.open_anomaly_view)
        self.status = QLabel("")
        self.status.setFont(QFont(self.font_family, 11))
        self.status.setStyleSheet("color: #333333;")
        self.status.setAlignment(Qt.AlignCenter)
        self.status_opacity = QGraphicsOpacityEffect()
        self.status.setGraphicsEffect(self.status_opacity)
        self.fade_anim = QPropertyAnimation(self.status_opacity, b"opacity")
        self.fade_anim.setDuration(500)
        right_layout.addSpacing(50)
        right_layout.addWidget(title)
        right_layout.addWidget(subtitle)
        right_layout.addSpacing(40)
        right_layout.addWidget(self.dump_button, alignment=Qt.AlignCenter)
        right_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)
        right_layout.addWidget(self.live_button, alignment=Qt.AlignCenter)
        right_layout.addWidget(self.view_button, alignment=Qt.AlignCenter)
        right_layout.addSpacing(30)
        right_layout.addWidget(self.status)
        right_layout.addStretch()
        layout.addLayout(right_layout)
        self.update_monitor_button_status()

    def run_task(self, func, message):
        self.loader = LoaderDialog(message, self)
        self.loader.show()
        self.thread = WorkerThread(func)
        self.thread.finished.connect(self.task_done)
        self.thread.start()

    @Slot(str)
    def task_done(self, result):
        self.loader.close()
        self.status.setText(result)
        self.fade_anim.stop()
        self.status_opacity.setOpacity(0.0)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()
        self.update_monitor_button_status()

    def open_anomaly_view(self):
        self.viewer = AnomalyViewer(self)
        self.viewer.showMaximized()

    def is_monitor_running(self):
        if os.path.exists(MONITOR_PID_FILE):
            try:
                with open(MONITOR_PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return True
            except (ValueError, OSError):
                os.remove(MONITOR_PID_FILE)
        return False

    def update_monitor_button_status(self):
        if self.is_monitor_running():
            self.live_button.setText("Live Monitor (Running)")
        else:
            self.live_button.setText("Live Monitor (Start)")

    def toggle_live_monitor(self):
        if self.is_monitor_running():
            try:
                with open(MONITOR_PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 15)
                time.sleep(1)
                if os.path.exists(MONITOR_PID_FILE):
                    os.remove(MONITOR_PID_FILE)
                self.status.setText("Live Monitor stopped.")
                logging.info("Live Monitor stopped from GUI.")
            except Exception as e:
                self.status.setText(f"Error stopping monitor: {e}")
        else:
            try:
                monitor_script_path = os.path.join(os.path.dirname(__file__), 'monitor.py')
                subprocess.Popen([sys.executable, monitor_script_path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 preexec_fn=os.setsid)
                self.status.setText("Live Monitor started.")
                logging.info("Live Monitor started from GUI.")
                time.sleep(2)
            except Exception as e:
                self.status.setText(f"Error starting monitor: {e}")
        self.update_monitor_button_status()

class WorkerThread(QThread):
    finished = Signal(str)
    def __init__(self, func):
        super().__init__()
        self.func = func
    def run(self):
        result = self.func()
        self.finished.emit(result)

class LoaderDialog(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc;")
        self.setFixedSize(300, 100)
        label = QLabel(message)
        label.setFont(QFont("Cantarell", 12))
        label.setStyleSheet("color: #222222;")
        label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    app.setWindowIcon(QIcon(icon_path))
    window = KomainuApp()
    setup_logging()
    window.show()
    sys.exit(app.exec())
