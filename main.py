import threading
import queue
import time
import math
import re
from scapy.all import sniff, Raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

packet_queue = queue.Queue()

def calculate_entropy(payload_bytes):
    if not payload_bytes:
        return 0
    byte_freq = [0] * 256
    for byte in payload_bytes:
        byte_freq[byte] += 1
    entropy = 0
    for freq in byte_freq:
        if freq > 0:
            prob = freq / len(payload_bytes)
            entropy -= prob * math.log2(prob)
    return entropy

def extract_features(packet):
    features = {}
    if packet.haslayer(IP):
        features['src_ip'] = packet[IP].src
        features['dst_ip'] = packet[IP].dst
        features['protocol'] = packet[IP].proto
    elif packet.haslayer(IPv6):
        features['src_ip'] = packet[IPv6].src
        features['dst_ip'] = packet[IPv6].dst
        features['protocol'] = packet[IPv6].nh
    else:
        features['src_ip'] = 'N/A'
        features['dst_ip'] = 'N/A'
        features['protocol'] = 'N/A'

    if packet.haslayer(TCP):
        features['src_port'] = packet[TCP].sport
        features['dst_port'] = packet[TCP].dport
    elif packet.haslayer(UDP):
        features['src_port'] = packet[UDP].sport
        features['dst_port'] = packet[UDP].dport
    else:
        features['src_port'] = 'N/A'
        features['dst_port'] = 'N/A'

    if packet.haslayer(Raw):
        payload = bytes(packet[Raw].load)
        features['payload_len'] = len(payload)
        features['payload_entropy'] = calculate_entropy(payload)
        features['payload_tokens'] = len(re.findall(rb'[a-zA-Z0-9]+', payload))
    else:
        features['payload_len'] = 0
        features['payload_entropy'] = 0
        features['payload_tokens'] = 0

    features['timestamp'] = time.time()
    return features

def packet_sniffer():
    print("[*] Packet sniffer started...")

    def process(packet):
        packet_queue.put(packet)
        print(f"[+] Packet captured: {packet.summary()}")

    sniff(prn=process, store=False)

def feature_extraction_worker():
    while True:
        packet = packet_queue.get()
        try:
            features = extract_features(packet)
            print("[*] Features extracted:", features)
        except Exception as e:
            print("[!] Feature extraction error:", e)
        finally:
            packet_queue.task_done()

if __name__ == "__main__":
    try:
        sniffer_thread = threading.Thread(target=packet_sniffer)
        extractor_thread = threading.Thread(target=feature_extraction_worker)

        sniffer_thread.daemon = True
        extractor_thread.daemon = True

        sniffer_thread.start()
        extractor_thread.start()

        sniffer_thread.join()
        extractor_thread.join()

    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")
