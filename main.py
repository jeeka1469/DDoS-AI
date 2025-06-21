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

logger = logging.getLogger("DDoSDetector")
logger.setLevel(logging.DEBUG) 

file_handler = logging.FileHandler("ddos_detection.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

model = joblib.load("unified_ddos_best_model.pkl")
scaler = joblib.load("unified_ddos_best_model_scaler.pkl")
metadata = joblib.load("unified_ddos_best_model_metadata.pkl")
label_encoders = metadata['label_encoders']
feature_columns = metadata['feature_columns']

flow_packets = defaultdict(deque)
src_ip_history = deque(maxlen=5000)
processing_queue = queue.Queue()
flow_lock = threading.Lock()

logger.info("Real-time DDoS flow prediction started")


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
        proto_enc = label_encoders['Protocol'].transform([flow_id[4]])[0]
    except:
        proto_enc = -1

    try:
        ip_enc = label_encoders['Source IP'].transform([flow_id[0]])[0]
    except:
        ip_enc = -1

    feat_dict = {
        'Flow Packets/s': packet_rate,
        'Packet Length Mean': mean_size,
        'Flow IAT Mean': mean_iat,
        'SYN Flag Count': syn_count,
        'ACK Flag Count': ack_count,
        'RST Flag Count': rst_count,
        'FIN Flag Count': fin_count,
        'Flow Duration': duration,
        'Flow Bytes/s': bytes_per_sec,
        'Down/Up Ratio': down_up_ratio,
        'Fwd Packets/s': fwd_rate,
        'Bwd Packets/s': bwd_rate,
        'Flow IAT Std': std_iat,
        'Source IP': ip_enc,
        'Protocol': proto_enc
    }

    return feat_dict


def process_packets():
    while True:
        packet = processing_queue.get()
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

                if not all(col in features for col in feature_columns):
                    logger.warning("Feature mismatch detected.")
                    continue

                X = np.array([[features[col] for col in feature_columns]], dtype=np.float32)
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
    processing_queue.put(packet)


def start_multi_interface_sniffing():
    interfaces_info = get_windows_if_list()
    sniffable_ifaces = [iface['name'] for iface in interfaces_info if iface['name'].startswith('\\Device\\NPF_')]

    logger.info(f"🌐 Sniffable interfaces detected: {sniffable_ifaces}")
    for iface in sniffable_ifaces:
        logger.info(f"Sniffing on interface: {iface}")
        threading.Thread(target=sniff, kwargs={
            'iface': iface,
            'prn': packet_handler,
            'store': False
        }, daemon=True).start()


if __name__ == '__main__':
    threading.Thread(target=process_packets, daemon=True).start()
    start_multi_interface_sniffing()
    while True:
        time.sleep(10)
