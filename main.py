import time
import threading
import math
import joblib
import numpy as np
from collections import defaultdict, deque, Counter
from scapy.all import sniff, IP, TCP, UDP
from scapy.arch.windows import get_windows_if_list
import queue
import logging
import sys
import signal
import os
from logging.handlers import RotatingFileHandler
import pandas as pd

MAX_QUEUE_SIZE = 10000
LOG_FILE = os.getenv("DDOS_LOG_FILE", "ddos_detection.log")
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
RUNNING = True

logger = logging.getLogger("DDoSDetector")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

model = joblib.load("unified_ddos_best_model.pkl")
scaler = joblib.load("unified_ddos_best_model_scaler.pkl")
metadata = joblib.load("unified_ddos_best_model_metadata.pkl")
label_encoders = metadata['label_encoders']
feature_columns = metadata['feature_columns']

flow_packets = defaultdict(deque)
src_ip_history = deque(maxlen=5000)
processing_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
flow_lock = threading.Lock()

def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received. Cleaning up and exiting...")
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def calculate_entropy(ip_list):
    if not ip_list:
        return 0.0
    counts = Counter(ip_list)
    total = len(ip_list)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def get_tcp_flags(tcp_layer):
    flags = tcp_layer.flags
    return {
        'SYN': int(flags & 0x02 != 0),
        'ACK': int(flags & 0x10 != 0),
        'FIN': int(flags & 0x01 != 0),
        'RST': int(flags & 0x04 != 0),
    }

def extract_features(flow_id, packets, entropy):
    times = [pkt[0] for pkt in packets]
    sizes = [pkt[1] for pkt in packets]
    flags_list = [pkt[2] for pkt in packets]
    directions = [pkt[3] for pkt in packets]

    duration = times[-1] - times[0] if len(times) > 1 else 0.001
    packet_rate = len(packets) / duration if duration > 0 else 0
    mean_size = np.mean(sizes)
    iats = [times[i] - times[i - 1] for i in range(1, len(times))]
    mean_iat = np.mean(iats) if iats else 0
    std_iat = np.std(iats) if iats else 0

    syn_count = sum(flag['SYN'] for flag in flags_list)
    ack_count = sum(flag['ACK'] for flag in flags_list)
    fin_count = sum(flag['FIN'] for flag in flags_list)
    rst_count = sum(flag['RST'] for flag in flags_list)

    total_bytes = sum(sizes)
    bytes_per_sec = total_bytes / duration if duration > 0 else 0

    fwd_count = directions.count('fwd')
    bwd_count = directions.count('bwd')
    fwd_rate = fwd_count / duration if duration > 0 else 0
    bwd_rate = bwd_count / duration if duration > 0 else 0
    down_up_ratio = bwd_count / fwd_count if fwd_count > 0 else bwd_count

    try:
        proto_enc = label_encoders['protocol'].transform([flow_id[4]])[0]
    except Exception:
        proto_enc = -1

    try:
        ip_enc = label_encoders['src_ip'].transform([flow_id[0]])[0]
    except Exception:
        ip_enc = -1

    feat_dict = {
        'flow_pkts_s': packet_rate,
        'pkt_len_mean': mean_size,
        'flow_iat_mean': mean_iat,
        'syn_flag_cnt': syn_count,
        'ack_flag_cnt': ack_count,
        'rst_flag_cnt': rst_count,
        'fin_flag_cnt': fin_count,
        'flow_duration': duration,
        'flow_byts_s': bytes_per_sec,
        'down_up_ratio': down_up_ratio,
        'fwd_pkts_s': fwd_rate,
        'bwd_pkts_s': bwd_rate,
        'flow_iat_std': std_iat,
        'src_ip': ip_enc,
        'protocol': proto_enc
    }

    return feat_dict

def process_packets():
    while RUNNING:
        try:
            packet = processing_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            if IP not in packet:
                continue

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            proto = packet[IP].proto
            src_port = dst_port = 0
            tcp_flags = {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0}

            if proto == 6 and TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                tcp_flags = get_tcp_flags(packet[TCP])
            elif proto == 17 and UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport

            flow_id = (src_ip, src_port, dst_ip, dst_port, proto)
            direction = 'fwd' if src_port < dst_port else 'bwd'
            timestamp = time.time()
            pkt_len = len(packet)

            with flow_lock:
                flow_packets[flow_id].append((timestamp, pkt_len, tcp_flags, direction))
                if len(flow_packets[flow_id]) > 50:
                    flow_packets[flow_id].popleft()
                src_ip_history.append(src_ip)

            if len(flow_packets[flow_id]) % 10 == 0:
                entropy = calculate_entropy(src_ip_history)
                features = extract_features(flow_id, list(flow_packets[flow_id]), entropy)
                
                missing_features = [col for col in feature_columns if col not in features]
                if missing_features:
                    logger.warning(f"Missing features for prediction: {missing_features}")
                    continue
                
                X = pd.DataFrame([[features[col] for col in feature_columns]], columns=feature_columns)
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                logger.info(f"[FLOW] {flow_id} => Predicted: {pred}")

            with flow_lock:
                now = time.time()
                stale_flows = [fid for fid, pkts in flow_packets.items() if now - pkts[-1][0] > 60]
                for fid in stale_flows:
                    del flow_packets[fid]

        except Exception as e:
            logger.error(f"Failed to process packet: {e}")
        finally:
            processing_queue.task_done()

def packet_handler(packet):
    try:
        logger.debug(f"Captured packet: {packet.summary()}")
        processing_queue.put(packet, block=False)
    except queue.Full:
        logger.warning("Processing queue full. Dropping packet.")

def choose_interface(interface_names):
    print("\nAvailable Network Interfaces:")
    for i, name in enumerate(interface_names):
        print(f"{i}: {name}")
    selected = input("Enter the index(es) of the interface(s) you want to sniff on (comma-separated): ")
    indexes = [int(i.strip()) for i in selected.split(",") if i.strip().isdigit()]
    return [interface_names[i] for i in indexes if 0 <= i < len(interface_names)]

def start_sniffing_on_selected_interfaces():
    interfaces_info = get_windows_if_list()
    interface_names = [iface['name'] for iface in interfaces_info]
    logger.info(f"Detected interfaces: {interface_names}")

    selected_interfaces = choose_interface(interface_names)
    if not selected_interfaces:
        logger.error("No valid interfaces selected. Exiting...")
        return

    for iface in selected_interfaces:
        logger.info(f"Starting sniffing on: {iface}")
        try:
            t = threading.Thread(target=sniff, kwargs={
                'iface': iface,
                'prn': packet_handler,
                'store': False,
                'stop_filter': lambda x: not RUNNING
            }, daemon=True)
            t.start()
        except Exception as e:
            logger.warning(f"Failed to start sniffing on {iface}: {e}")

def check_admin_permissions():
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if not is_admin:
            logger.warning("You are not running this script as Administrator. Packet capture may fail.")
        else:
            logger.info("Administrator privileges confirmed.")
    except Exception:
        logger.warning("Unable to check admin privileges.")

if __name__ == '__main__':
    check_admin_permissions()
    threading.Thread(target=process_packets, daemon=True).start()
    start_sniffing_on_selected_interfaces()
    try:
        while RUNNING:
            time.sleep(10)
    except KeyboardInterrupt:
        RUNNING = False
        logger.info("Keyboard interrupt received. Exiting...")
