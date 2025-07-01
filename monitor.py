import os
import sys
import json
import subprocess
import logging
import time
import joblib
import numpy as np
import scipy.sparse
from plyer import notification
from transformers import pipeline

BASE_DIR = os.path.expanduser("~/komainud")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
MONITOR_LOG_FILE = os.path.join(LOG_DIR, "komainu_monitor.log")

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
PID_FILE = os.path.join(BASE_DIR, "komainu_monitor.pid")

def setup_monitoring_logging():
    logging.basicConfig(filename=MONITOR_LOG_FILE,
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        force=True)

def send_notification(title, message):
    try:
        notification.notify(
            title=title,
            message=message,
            app_name='Komainu Monitor',
            timeout=10
        )
        logging.info(f"Notification sent: {title} - {message}")
    except Exception as e:
        logging.warning(f"Failed to send notification: {e}")

def daemonize():
    if os.fork() > 0:
        exit(0)
    os.setsid()
    if os.fork() > 0:
        exit(0)
    os.umask(0)
    os.chdir('/')
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def load_models():
    logging.info("Loading machine learning models...")
    try:
        isolation_forest = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.joblib'))
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
        hashing_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'hashing_vectorizer.joblib'))
        logging.info("Models loaded successfully.")
        return isolation_forest, kmeans, hashing_vectorizer
    except FileNotFoundError:
        logging.error("Models not found. Please run training first.")
        send_notification("Komainu Monitor Error", "Models not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        send_notification("Komainu Monitor Error", f"Failed to load models: {e}. Exiting.")
        sys.exit(1)

def extract_features_single(log_entry, vectorizer):
    timestamp_str = log_entry.get('__REALTIME_TIMESTAMP')
    try:
        timestamp_sec = int(timestamp_str) / 1_000_000_000 if timestamp_str else time.time()
    except ValueError:
        timestamp_sec = time.time()

    timestamp_arr = np.array([[timestamp_sec]], dtype=float)

    message = log_entry.get('MESSAGE', '')
    if not isinstance(message, str):
        message = str(message)

    X_msg = vectorizer.transform([message])

    prio_str = log_entry.get('PRIORITY', '5')
    try:
        prio = int(prio_str)
    except ValueError:
        prio = 5
    priority_arr = np.array([[prio]], dtype=float)

    X_combined = scipy.sparse.hstack([timestamp_arr, priority_arr, X_msg])
    return X_combined, X_msg, message, prio_str, timestamp_str

summarizer = pipeline("summarization", model="google/flan-t5-small")

def summarize_anomaly(message, priority, timestamp):
    prompt = f"""
You are a Linux system expert. Explain this system log anomaly in 4-5 sentences (about 40 words).
Priority: {priority}
Timestamp: {timestamp}
Message: {message}
Explanation:"""
    try:
        summary = summarizer(prompt, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        logging.error(f"Summarizer failed: {e}")
        return "Summary unavailable"

def monitor_logs():
    setup_monitoring_logging()
    logging.info("Komainu Monitor Daemon started.")
    send_notification("Komainu Monitor", "Started successfully. Monitoring logs...")

    isolation_forest, kmeans, hashing_vectorizer = load_models()
    logging.info("Generative summarizer (Flan-T5) loaded.")

    cmd = ['journalctl', '-f', '-o', 'json', '--no-pager']
    logging.info(f"Starting journalctl with command: {' '.join(cmd)}")

    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_entry = json.loads(line)
                    X_combined, X_msg, message, priority, timestamp = extract_features_single(log_entry, hashing_vectorizer)
                    anomaly_pred = isolation_forest.predict(X_combined)

                    if anomaly_pred == -1:
                        cluster_pred = kmeans.predict(X_msg)
                        summary = summarize_anomaly(message, priority, timestamp)

                        record = {
                            "timestamp": timestamp,
                            "priority": priority,
                            "cluster": int(cluster_pred[0]),
                            "message": message[:500],
                            "summary": summary
                        }

                        logging.info("ANOMALY: " + json.dumps(record))
                        send_notification("Komainu Anomaly Alert", f"Anomaly (Cluster {record['cluster']}): {record['message'][:120]}...")

                except json.JSONDecodeError:
                    logging.warning(f"Failed to decode journal line: {line[:100]}...")
                except Exception as e:
                    logging.error(f"Error processing log: {e} - Line: {line[:100]}...")

            stderr_output = proc.stderr.read()
            if stderr_output:
                logging.error(f"journalctl stderr: {stderr_output}")

    except FileNotFoundError:
        logging.critical("'journalctl' command not found.")
        send_notification("Komainu Monitor Error", "'journalctl' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Monitoring loop error: {e}")
        send_notification("Komainu Monitor Critical Error", f"Monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            print(f"Komainu Monitor is already running with PID {pid}.")
            sys.exit(0)
        except (ValueError, OSError):
            os.remove(PID_FILE)
            print("Stale PID file found. Starting fresh daemon.")
    daemonize()
    monitor_logs()
