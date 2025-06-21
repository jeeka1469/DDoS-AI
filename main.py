import time
import threading
import math
import joblib
import numpy as np
from collections import defaultdict, deque, Counter
from scapy.all import sniff, IP, TCP, UDP

# Load model, scaler, and metadata
model = joblib.load("unified_ddos_best_model.pkl")
scaler = joblib.load("unified_ddos_best_model_scaler.pkl")
metadata = joblib.load("unified_ddos_best_model_metadata.pkl")
label_encoders = metadata['label_encoders']
feature_columns = metadata['feature_columns']

flow_packets = defaultdict(deque)
src_ip_history = deque(maxlen=5000)

print("\U0001F50D Real-time DDoS flow prediction started...")

def calculate_entropy(ip_list):
    if not ip_list:
        return 0.0
    counts = Counter(ip_list)
    total = len(ip_list)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

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

def packet_handler(packet):
    if IP not in packet:
        return

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    proto = packet[IP].proto
    src_port = dst_port = 0
    tcp_flags = {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0}

    direction = 'fwd'

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
    flow_packets[flow_id].append((timestamp, pkt_len, tcp_flags, direction))

    if len(flow_packets[flow_id]) > 50:
        flow_packets[flow_id].popleft()

    src_ip_history.append(src_ip)

    if len(flow_packets[flow_id]) % 10 == 0:
        entropy = calculate_entropy(src_ip_history)
        try:
            features = extract_features(flow_id, flow_packets[flow_id], entropy)
            X = np.array([[features[col] for col in feature_columns]], dtype=np.float32)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            print(f"\n[FLOW] {flow_id} => \U0001F6E1\ufe0f Predicted: {pred}")
        except Exception as e:
            print(f"[WARN] Skipping prediction due to error: {e}")

def start_sniff(interface=None):
    sniff(prn=packet_handler, iface=interface, store=False)

if __name__ == '__main__':
    start_sniff(interface=None)
