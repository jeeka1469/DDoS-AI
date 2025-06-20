#!/usr/bin/env python3
"""
Enhanced Network Intrusion Detection System
Integrated with Unified DDoS Model
Works with model.pkl, scaler.pkl, and metadata.pkl
"""

import pickle
import numpy as np
import pandas as pd
import threading
import queue
import time
import logging
import socket
import struct
import os
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    print("Warning: Scapy not installed. Install with: pip install scapy")
    SCAPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_nids.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NetworkFlowTracker:
    """Tracks network flows and calculates flow-based features"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'end_time': None,
            'fwd_packets': [],
            'bwd_packets': [],
            'flags': defaultdict(int),
            'bytes_sent': 0,
            'bytes_received': 0
        })
        self.flow_timeout = 120  # 2 minutes timeout
        
    def get_flow_id(self, packet):
        """Generate flow ID from packet"""
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                protocol = 6
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol = 17
            else:
                src_port = dst_port = 0
                protocol = packet[IP].proto
                
            # Create bidirectional flow ID
            flow_id = tuple(sorted([
                (src_ip, src_port, dst_ip, dst_port, protocol),
                (dst_ip, dst_port, src_ip, src_port, protocol)
            ])[0])
            
            return flow_id, (src_ip == flow_id[0])
        return None, None
    
    def update_flow(self, packet, timestamp):
        """Update flow information with new packet"""
        flow_id, is_forward = self.get_flow_id(packet)
        if not flow_id:
            return None
            
        flow = self.flows[flow_id]
        
        if flow['start_time'] is None:
            flow['start_time'] = timestamp
            
        flow['end_time'] = timestamp
        flow['packets'].append((packet, timestamp, is_forward))
        
        if is_forward:
            flow['fwd_packets'].append((packet, timestamp))
        else:
            flow['bwd_packets'].append((packet, timestamp))
            
        # Update flags if TCP
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            if tcp_flags & 0x01: flow['flags']['FIN'] += 1
            if tcp_flags & 0x02: flow['flags']['SYN'] += 1
            if tcp_flags & 0x04: flow['flags']['RST'] += 1
            if tcp_flags & 0x08: flow['flags']['PSH'] += 1
            if tcp_flags & 0x10: flow['flags']['ACK'] += 1
            if tcp_flags & 0x20: flow['flags']['URG'] += 1
            if tcp_flags & 0x40: flow['flags']['ECE'] += 1
            if tcp_flags & 0x80: flow['flags']['CWR'] += 1
            
        return flow_id

class UnifiedFeatureExtractor:
    """Enhanced feature extractor that matches the unified model's expected features"""
    
    def __init__(self, feature_columns=None):
        self.flow_tracker = NetworkFlowTracker()
        self.feature_columns = feature_columns
        self.expected_features = len(feature_columns) if feature_columns else 87
        
    def safe_divide(self, a, b):
        """Safe division to avoid division by zero"""
        return a / b if b != 0 else 0
    
    def calculate_packet_lengths(self, packets):
        """Calculate packet length statistics"""
        if not packets:
            return [0] * 4
            
        lengths = [len(pkt[0]) for pkt in packets]
        return [
            max(lengths) if lengths else 0,
            min(lengths) if lengths else 0,
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0
        ]
    
    def calculate_iat_stats(self, packets):
        """Calculate Inter-Arrival Time statistics"""
        if len(packets) < 2:
            return [0] * 5
            
        iats = []
        for i in range(1, len(packets)):
            iat = packets[i][1] - packets[i-1][1]
            iats.append(iat * 1000000)  # Convert to microseconds
            
        if not iats:
            return [0] * 5
            
        return [
            sum(iats),
            np.mean(iats),
            np.std(iats),
            max(iats),
            min(iats)
        ]
    
    def extract_unified_features(self, flow_id, flow_data):
        """Extract features matching the unified model's training data structure"""
        try:
            # Basic flow information
            total_packets = len(flow_data['packets'])
            fwd_packets = len(flow_data['fwd_packets'])
            bwd_packets = len(flow_data['bwd_packets'])
            
            duration = 0
            if flow_data['end_time'] and flow_data['start_time']:
                duration = (flow_data['end_time'] - flow_data['start_time']) * 1000000
            
            # Packet lengths
            all_lengths = [len(pkt[0]) for pkt in flow_data['packets']]
            fwd_lengths = [len(pkt[0]) for pkt in flow_data['fwd_packets']]
            bwd_lengths = [len(pkt[0]) for pkt in flow_data['bwd_packets']]
            
            # Initialize feature dictionary to match training data
            features_dict = {}
            
            # Create comprehensive feature set based on typical network flow features
            # These should match the columns from your training data
            
            # Basic flow stats
            features_dict['Flow Duration'] = duration
            features_dict['Total Fwd Packets'] = fwd_packets
            features_dict['Total Backward Packets'] = bwd_packets
            features_dict['Total Length of Fwd Packets'] = sum(fwd_lengths) if fwd_lengths else 0
            features_dict['Total Length of Bwd Packets'] = sum(bwd_lengths) if bwd_lengths else 0
            
            # Packet length statistics
            if fwd_lengths:
                features_dict['Fwd Packet Length Max'] = max(fwd_lengths)
                features_dict['Fwd Packet Length Min'] = min(fwd_lengths)
                features_dict['Fwd Packet Length Mean'] = np.mean(fwd_lengths)
                features_dict['Fwd Packet Length Std'] = np.std(fwd_lengths)
            else:
                features_dict['Fwd Packet Length Max'] = 0
                features_dict['Fwd Packet Length Min'] = 0
                features_dict['Fwd Packet Length Mean'] = 0
                features_dict['Fwd Packet Length Std'] = 0
                
            if bwd_lengths:
                features_dict['Bwd Packet Length Max'] = max(bwd_lengths)
                features_dict['Bwd Packet Length Min'] = min(bwd_lengths)
                features_dict['Bwd Packet Length Mean'] = np.mean(bwd_lengths)
                features_dict['Bwd Packet Length Std'] = np.std(bwd_lengths)
            else:
                features_dict['Bwd Packet Length Max'] = 0
                features_dict['Bwd Packet Length Min'] = 0
                features_dict['Bwd Packet Length Mean'] = 0
                features_dict['Bwd Packet Length Std'] = 0
            
            # Flow rates
            total_bytes = sum(all_lengths)
            features_dict['Flow Bytes/s'] = self.safe_divide(total_bytes, duration / 1000000) if duration > 0 else 0
            features_dict['Flow Packets/s'] = self.safe_divide(total_packets, duration / 1000000) if duration > 0 else 0
            
            # Inter-arrival time statistics
            flow_iat = self.calculate_iat_stats(flow_data['packets'])
            features_dict['Flow IAT Mean'] = flow_iat[1] if len(flow_iat) > 1 else 0
            features_dict['Flow IAT Std'] = flow_iat[2] if len(flow_iat) > 2 else 0
            features_dict['Flow IAT Max'] = flow_iat[3] if len(flow_iat) > 3 else 0
            features_dict['Flow IAT Min'] = flow_iat[4] if len(flow_iat) > 4 else 0
            
            fwd_iat = self.calculate_iat_stats(flow_data['fwd_packets'])
            features_dict['Fwd IAT Total'] = fwd_iat[0] if len(fwd_iat) > 0 else 0
            features_dict['Fwd IAT Mean'] = fwd_iat[1] if len(fwd_iat) > 1 else 0
            features_dict['Fwd IAT Std'] = fwd_iat[2] if len(fwd_iat) > 2 else 0
            features_dict['Fwd IAT Max'] = fwd_iat[3] if len(fwd_iat) > 3 else 0
            features_dict['Fwd IAT Min'] = fwd_iat[4] if len(fwd_iat) > 4 else 0
            
            bwd_iat = self.calculate_iat_stats(flow_data['bwd_packets'])
            features_dict['Bwd IAT Total'] = bwd_iat[0] if len(bwd_iat) > 0 else 0
            features_dict['Bwd IAT Mean'] = bwd_iat[1] if len(bwd_iat) > 1 else 0
            features_dict['Bwd IAT Std'] = bwd_iat[2] if len(bwd_iat) > 2 else 0
            features_dict['Bwd IAT Max'] = bwd_iat[3] if len(bwd_iat) > 3 else 0
            features_dict['Bwd IAT Min'] = bwd_iat[4] if len(bwd_iat) > 4 else 0
            
            # TCP Flags
            features_dict['Fwd PSH Flags'] = flow_data['flags']['PSH']
            features_dict['Bwd PSH Flags'] = 0  # Simplified
            features_dict['Fwd URG Flags'] = flow_data['flags']['URG']
            features_dict['Bwd URG Flags'] = 0  # Simplified
            
            # Header lengths
            features_dict['Fwd Header Length'] = fwd_packets * 20
            features_dict['Bwd Header Length'] = bwd_packets * 20
            
            # Packets per second
            features_dict['Fwd Packets/s'] = self.safe_divide(fwd_packets, duration / 1000000) if duration > 0 else 0
            features_dict['Bwd Packets/s'] = self.safe_divide(bwd_packets, duration / 1000000) if duration > 0 else 0
            
            # Packet size statistics
            if all_lengths:
                features_dict['Min Packet Length'] = min(all_lengths)
                features_dict['Max Packet Length'] = max(all_lengths)
                features_dict['Packet Length Mean'] = np.mean(all_lengths)
                features_dict['Packet Length Std'] = np.std(all_lengths)
                features_dict['Packet Length Variance'] = np.var(all_lengths)
            else:
                features_dict['Min Packet Length'] = 0
                features_dict['Max Packet Length'] = 0
                features_dict['Packet Length Mean'] = 0
                features_dict['Packet Length Std'] = 0
                features_dict['Packet Length Variance'] = 0
            
            # TCP Flag counts
            features_dict['FIN Flag Count'] = flow_data['flags']['FIN']
            features_dict['SYN Flag Count'] = flow_data['flags']['SYN']
            features_dict['RST Flag Count'] = flow_data['flags']['RST']
            features_dict['PSH Flag Count'] = flow_data['flags']['PSH']
            features_dict['ACK Flag Count'] = flow_data['flags']['ACK']
            features_dict['URG Flag Count'] = flow_data['flags']['URG']
            features_dict['CWR Flag Count'] = flow_data['flags']['CWR']
            features_dict['ECE Flag Count'] = flow_data['flags']['ECE']
            
            # Additional derived features
            features_dict['Down/Up Ratio'] = self.safe_divide(bwd_packets, fwd_packets) if fwd_packets > 0 else 0
            features_dict['Average Packet Size'] = np.mean(all_lengths) if all_lengths else 0
            features_dict['Avg Fwd Segment Size'] = np.mean(fwd_lengths) if fwd_lengths else 0
            features_dict['Avg Bwd Segment Size'] = np.mean(bwd_lengths) if bwd_lengths else 0
            
            # Subflow features
            features_dict['Subflow Fwd Packets'] = fwd_packets
            features_dict['Subflow Fwd Bytes'] = sum(fwd_lengths) if fwd_lengths else 0
            features_dict['Subflow Bwd Packets'] = bwd_packets
            features_dict['Subflow Bwd Bytes'] = sum(bwd_lengths) if bwd_lengths else 0
            
            # Additional network features
            features_dict['Init_Win_bytes_forward'] = 8192  # Default TCP window size
            features_dict['Init_Win_bytes_backward'] = 8192
            features_dict['act_data_pkt_fwd'] = fwd_packets
            features_dict['min_seg_size_forward'] = min(fwd_lengths) if fwd_lengths else 0
            
            # Active/Idle time features
            features_dict['Active Mean'] = duration / 1000000 if duration > 0 else 0
            features_dict['Active Std'] = 0
            features_dict['Active Max'] = duration / 1000000 if duration > 0 else 0
            features_dict['Active Min'] = 0
            features_dict['Idle Mean'] = 0
            features_dict['Idle Std'] = 0
            features_dict['Idle Max'] = 0
            features_dict['Idle Min'] = 0
            
            # Protocol-specific features
            features_dict['SimillarHTTP'] = 0
            features_dict['Inbound'] = 1 if bwd_packets > 0 else 0
            
            # Convert to feature array matching the training data structure
            if self.feature_columns:
                # Use the exact feature columns from training
                feature_array = []
                for col in self.feature_columns:
                    if col in features_dict:
                        feature_array.append(features_dict[col])
                    else:
                        # Fill missing features with 0
                        feature_array.append(0.0)
                        
                return np.array(feature_array, dtype=np.float32)
            else:
                # Fallback: return all features as array
                return np.array(list(features_dict.values()), dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting unified features: {e}")
            return None

class UnifiedNetworkIDSModel:
    """Enhanced Network IDS Model Handler with Unified Model Integration"""
    
    def __init__(self, model_path='unified_ddos_best_model.pkl', 
                 scaler_path='unified_ddos_best_model_scaler.pkl',
                 metadata_path='unified_ddos_best_model_metadata.pkl'):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.n_features = None
        
        self.load_unified_models(model_path, scaler_path, metadata_path)
        
    def load_unified_models(self, model_path, scaler_path, metadata_path):
        """Load unified model, scaler, and metadata"""
        try:
            logger.info("Loading unified DDoS detection models...")
            
            # Check if files exist
            required_files = [model_path, scaler_path, metadata_path]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
            
            # Load metadata first
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.feature_columns = metadata.get('feature_columns', [])
                self.label_encoders = metadata.get('label_encoders', {})
                self.n_features = metadata.get('n_features', len(self.feature_columns))
                
                logger.info(f"Metadata loaded successfully")
                logger.info(f"Expected features: {self.n_features}")
                logger.info(f"Feature columns: {len(self.feature_columns)}")
                logger.info(f"Label encoders: {list(self.label_encoders.keys())}")
                
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_path}: {e}")
                raise
            
            # Load model
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Unified model loaded successfully")
                logger.info(f"Model type: {type(self.model)}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                raise
            
            # Load scaler
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
                logger.info(f"Scaler type: {type(self.scaler)}")
            except Exception as e:
                logger.error(f"Error loading scaler from {scaler_path}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error loading unified models: {e}")
            raise
    
    def predict(self, features_array):
        """Predict attack type from features array using unified model"""
        try:
            # Ensure features is 2D array
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            # Check feature count
            if features_array.shape[1] != self.n_features:
                logger.warning(f"Feature count mismatch. Expected: {self.n_features}, Got: {features_array.shape[1]}")
                # Pad or truncate as necessary
                if features_array.shape[1] < self.n_features:
                    padding = np.zeros((features_array.shape[0], self.n_features - features_array.shape[1]))
                    features_array = np.concatenate([features_array, padding], axis=1)
                else:
                    features_array = features_array[:, :self.n_features]
            
            # Handle infinite and NaN values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Scale features
            X_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(X_scaled)[0]
                confidence = max(prediction_proba)
            else:
                confidence = 1.0
            
            return {
                'prediction': str(prediction),
                'confidence': confidence,
                'raw_prediction': prediction,
                'model_type': 'unified_ddos'
            }
            
        except Exception as e:
            logger.error(f"Error in unified model prediction: {e}")
            return None

class UnifiedNetworkIDSMain:
    """Enhanced Network Intrusion Detection System with Unified Model"""
    
    def __init__(self):
        self.packet_queue = queue.Queue(maxsize=1000)
        self.feature_queue = queue.Queue(maxsize=100)
        self.prediction_queue = queue.Queue(maxsize=100)
        
        self.feature_extractor = None
        self.ml_model = None
        self.running = False
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'features_extracted': 0,
            'predictions_made': 0,
            'attacks_detected': 0,
            'start_time': None
        }
        
    def initialize_unified_model(self):
        """Initialize unified ML model and feature extractor"""
        try:
            # Initialize model
            self.ml_model = UnifiedNetworkIDSModel()
            
            # Initialize feature extractor with model's feature columns
            self.feature_extractor = UnifiedFeatureExtractor(
                feature_columns=self.ml_model.feature_columns
            )
            
            logger.info("Unified ML model and feature extractor initialized successfully")
            logger.info(f"Model expects {self.ml_model.n_features} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize unified model: {e}")
            return False
    
    def packet_capture_thread(self):
        """Thread 1: Capture packets from network"""
        def packet_handler(packet):
            if self.running:
                try:
                    timestamp = time.time()
                    if not self.packet_queue.full():
                        self.packet_queue.put((packet, timestamp))
                        self.stats['packets_processed'] += 1
                except Exception as e:
                    logger.error(f"Error in packet handler: {e}")
        
        try:
            if SCAPY_AVAILABLE:
                logger.info("Starting enhanced packet capture...")
                sniff(prn=packet_handler, store=0, stop_filter=lambda x: not self.running)
            else:
                logger.error("Scapy not available, cannot capture packets")
                
        except Exception as e:
            logger.error(f"Error in packet capture: {e}")
    
    def feature_extraction_thread(self):
        """Thread 2: Extract features using unified feature extractor"""
        last_cleanup = time.time()
        last_stats = time.time()
        
        while self.running:
            try:
                # Get packet from queue
                packet, timestamp = self.packet_queue.get(timeout=1.0)
                
                # Update flow tracker
                flow_id = self.feature_extractor.flow_tracker.update_flow(packet, timestamp)
                
                if flow_id:
                    flow_data = self.feature_extractor.flow_tracker.flows[flow_id]
                    
                    # Extract features if flow has enough packets
                    if len(flow_data['packets']) >= 10:  # Minimum packets for stable analysis
                        features = self.feature_extractor.extract_unified_features(flow_id, flow_data)
                        if features is not None and not self.feature_queue.full():
                            self.feature_queue.put((flow_id, features))
                            self.stats['features_extracted'] += 1
                
                # Cleanup old flows every 60 seconds
                if time.time() - last_cleanup > 60:
                    self.cleanup_old_flows()
                    last_cleanup = time.time()
                
                # Print stats every 30 seconds
                if time.time() - last_stats > 30:
                    self.print_stats()
                    last_stats = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in unified feature extraction: {e}")
    
    def ml_inference_thread(self):
        """Thread 3: Perform ML inference using unified model"""
        while self.running:
            try:
                flow_id, features = self.feature_queue.get(timeout=1.0)
                
                if self.ml_model:
                    prediction = self.ml_model.predict(features)
                    if prediction and not self.prediction_queue.full():
                        self.prediction_queue.put((flow_id, features, prediction))
                        self.stats['predictions_made'] += 1
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in unified ML inference: {e}")
    
    def logging_alert_thread(self):
        """Thread 4: Enhanced logging and alerting with unified model results"""
        while self.running:
            try:
                flow_id, features, prediction = self.prediction_queue.get(timeout=1.0)
                
                # Enhanced logging with unified model information
                log_message = (f"Flow {str(flow_id)[:50]}...: "
                             f"{prediction['prediction']} "
                             f"(confidence: {prediction['confidence']:.3f}) "
                             f"[{prediction['model_type']}]")
                
                # Check if this is an attack
                is_attack = (prediction['prediction'] != 'BENIGN' and 
                           'BENIGN' not in prediction['prediction'].upper())
                
                if is_attack:
                    self.stats['attacks_detected'] += 1
                    logger.warning(f" ATTACK DETECTED: {log_message}")
                    
                    # Additional detailed logging for attacks
                    logger.warning(f"   Flow details: {flow_id}")
                    logger.warning(f"   Feature vector shape: {features.shape}")
                    logger.warning(f"   Raw prediction: {prediction.get('raw_prediction', 'N/A')}")
                    
                else:
                    logger.info(f" Normal traffic: {log_message}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in enhanced logging/alerting: {e}")
    
    def cleanup_old_flows(self):
        """Clean up old flows to prevent memory buildup"""
        current_time = time.time()
        flows_to_remove = []
        
        for flow_id, flow_data in self.feature_extractor.flow_tracker.flows.items():
            if flow_data['end_time'] and (current_time - flow_data['end_time']) > 300:  # 5 minutes
                flows_to_remove.append(flow_id)
        
        for flow_id in flows_to_remove:
            del self.feature_extractor.flow_tracker.flows[flow_id]
            
        if flows_to_remove:
            logger.info(f" Cleaned up {len(flows_to_remove)} old flows")
    
    def print_stats(self):
        """Print system statistics"""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            logger.info(f" Stats - Runtime: {runtime:.1f}s, "
                       f"Packets: {self.stats['packets_processed']}, "
                       f"Features: {self.stats['features_extracted']}, "
                       f"Predictions: {self.stats['predictions_made']}, "
                       f"Attacks: {self.stats['attacks_detected']}")
    
    def start(self):
        """Start the Enhanced Network IDS with Unified Model"""
        logger.info(" Starting Enhanced Network Intrusion Detection System with Unified Model...")
        
        # Initialize unified model
        if not self.initialize_unified_model():
            logger.error(" Failed to initialize unified model. Exiting.")
            return False
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start all threads
        threads = [
            threading.Thread(target=self.packet_capture_thread, name="UnifiedPacketCapture"),
            threading.Thread(target=self.feature_extraction_thread, name="UnifiedFeatureExtraction"),
            threading.Thread(target=self.ml_inference_thread, name="UnifiedMLInference"),
            threading.Thread(target=self.logging_alert_thread, name="UnifiedLoggingAlert")
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            logger.info(f" Started {thread.name} thread")
        
        try:
            # Keep main thread alive
            logger.info(" Enhanced NIDS with Unified Model is running... Press Ctrl+C to stop")
            logger.info(f" Monitoring with {self.ml_model.n_features} features per flow")
            
            while self.running:
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("  Received interrupt signal, shutting down...")
            self.running = False
            
            # Print final stats
            self.print_stats()
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=5)
            
            logger.info(" Enhanced Network IDS stopped")
            
        return True

def main():
    """Main function for Enhanced Network IDS"""
    try:
        logger.info("🔧 Initializing Enhanced Network IDS with Unified DDoS Model...")
        nids = UnifiedNetworkIDSMain()
        nids.start()
        
    except Exception as e:
        logger.error(f" Critical error in main: {e}")
        raise

if __name__ == "__main__":
    main()