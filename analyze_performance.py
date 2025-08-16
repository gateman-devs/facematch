#!/usr/bin/env python3
"""
Performance Analysis Script for Video Liveness Processing
Analyzes performance logs to determine if the system is performing well enough.
"""

import re
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse
import os

class PerformanceAnalyzer:
    def __init__(self, log_file_path: str = "logs/facematch.log"):
        self.log_file_path = log_file_path
        self.performance_data = []
        self.session_data = {}
        
    def parse_log_file(self) -> None:
        """Parse the performance log file and extract timing data."""
        if not os.path.exists(self.log_file_path):
            print(f"Warning: Log file {self.log_file_path} not found.")
            return
            
        with open(self.log_file_path, 'r') as f:
            for line in f:
                self._parse_log_line(line.strip())
    
    def _parse_log_line(self, line: str) -> None:
        """Parse a single log line and extract performance data."""
        # Parse timestamp and log level
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)', line)
        if not timestamp_match:
            return
            
        timestamp_str, level, message = timestamp_match.groups()
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        
        # Parse different types of performance messages
        if 'VIDEO_LIVENESS_REQUEST_START' in message:
            self._parse_request_start(timestamp, message)
        elif 'VIDEO_FILE_RECEIVE_COMPLETE' in message:
            self._parse_file_receive(timestamp, message)
        elif 'VIDEO_PROCESSING_COMPLETE' in message:
            self._parse_processing_complete(timestamp, message)
        elif 'VIDEO_LIVENESS_COMPLETE' in message:
            self._parse_liveness_complete(timestamp, message)
        elif 'VIDEO_LIVENESS_FAILED' in message:
            self._parse_liveness_failed(timestamp, message)
    
    def _parse_request_start(self, timestamp: datetime, message: str) -> None:
        """Parse request start message."""
        session_match = re.search(r'session_id: ([^,]+)', message)
        challenge_match = re.search(r'challenge_type: ([^,]+)', message)
        
        if session_match:
            session_id = session_match.group(1)
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    'request_start': timestamp,
                    'challenge_type': challenge_match.group(1) if challenge_match else 'unknown'
                }
    
    def _parse_file_receive(self, timestamp: datetime, message: str) -> None:
        """Parse file receive completion message."""
        session_match = re.search(r'session_id: ([^,]+)', message)
        duration_match = re.search(r'duration: ([\d.]+)s', message)
        size_match = re.search(r'size: (\d+) bytes', message)
        
        if session_match and duration_match:
            session_id = session_match.group(1)
            if session_id in self.session_data:
                self.session_data[session_id]['file_receive_time'] = float(duration_match.group(1))
                self.session_data[session_id]['file_size'] = int(size_match.group(1)) if size_match else 0
    
    def _parse_processing_complete(self, timestamp: datetime, message: str) -> None:
        """Parse processing completion message."""
        session_match = re.search(r'session_id: ([^,]+)', message)
        duration_match = re.search(r'duration: ([\d.]+)s', message)
        success_match = re.search(r'success: (\w+)', message)
        
        if session_match and duration_match:
            session_id = session_match.group(1)
            if session_id in self.session_data:
                self.session_data[session_id]['processing_time'] = float(duration_match.group(1))
                self.session_data[session_id]['processing_success'] = success_match.group(1) == 'True' if success_match else False
    
    def _parse_liveness_complete(self, timestamp: datetime, message: str) -> None:
        """Parse liveness completion message."""
        session_match = re.search(r'session_id: ([^,]+)', message)
        result_match = re.search(r'result: (\w+)', message)
        total_time_match = re.search(r'total_time: ([\d.]+)s', message)
        processing_time_match = re.search(r'processing_time: ([\d.]+)s', message)
        receive_time_match = re.search(r'video_receive_time: ([\d.]+)s', message)
        
        if session_match and total_time_match:
            session_id = session_match.group(1)
            if session_id in self.session_data:
                self.session_data[session_id].update({
                    'completion_time': timestamp,
                    'result': result_match.group(1) if result_match else 'unknown',
                    'total_time': float(total_time_match.group(1)),
                    'processing_time': float(processing_time_match.group(1)) if processing_time_match else 0,
                    'receive_time': float(receive_time_match.group(1)) if receive_time_match else 0,
                    'status': 'completed'
                })
    
    def _parse_liveness_failed(self, timestamp: datetime, message: str) -> None:
        """Parse liveness failure message."""
        session_match = re.search(r'session_id: ([^,]+)', message)
        total_time_match = re.search(r'total_time: ([\d.]+)s', message)
        error_match = re.search(r'error: ([^,]+)', message)
        
        if session_match and total_time_match:
            session_id = session_match.group(1)
            if session_id in self.session_data:
                self.session_data[session_id].update({
                    'completion_time': timestamp,
                    'total_time': float(total_time_match.group(1)),
                    'error': error_match.group(1) if error_match else 'Unknown error',
                    'status': 'failed'
                })
    
    def analyze_performance(self) -> Dict:
        """Analyze performance data and return insights."""
        if not self.session_data:
            return {"error": "No performance data found"}
        
        completed_sessions = [s for s in self.session_data.values() if s.get('status') == 'completed']
        failed_sessions = [s for s in self.session_data.values() if s.get('status') == 'failed']
        
        analysis = {
            'total_sessions': len(self.session_data),
            'completed_sessions': len(completed_sessions),
            'failed_sessions': len(failed_sessions),
            'success_rate': len(completed_sessions) / len(self.session_data) if self.session_data else 0,
            'timing_analysis': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        if completed_sessions:
            # Timing analysis
            total_times = [s['total_time'] for s in completed_sessions]
            processing_times = [s.get('processing_time', 0) for s in completed_sessions]
            receive_times = [s.get('receive_time', 0) for s in completed_sessions]
            
            analysis['timing_analysis'] = {
                'total_time': {
                    'mean': statistics.mean(total_times),
                    'median': statistics.median(total_times),
                    'min': min(total_times),
                    'max': max(total_times),
                    'p95': sorted(total_times)[int(len(total_times) * 0.95)] if len(total_times) > 0 else 0,
                    'p99': sorted(total_times)[int(len(total_times) * 0.99)] if len(total_times) > 0 else 0
                },
                'processing_time': {
                    'mean': statistics.mean(processing_times),
                    'median': statistics.median(processing_times),
                    'min': min(processing_times),
                    'max': max(processing_times)
                },
                'receive_time': {
                    'mean': statistics.mean(receive_times),
                    'median': statistics.median(receive_times),
                    'min': min(receive_times),
                    'max': max(receive_times)
                }
            }
            
            # Performance metrics
            avg_total_time = statistics.mean(total_times)
            avg_processing_time = statistics.mean(processing_times)
            
            analysis['performance_metrics'] = {
                'avg_total_time': avg_total_time,
                'avg_processing_time': avg_processing_time,
                'processing_efficiency': avg_processing_time / avg_total_time if avg_total_time > 0 else 0,
                'throughput': 1 / avg_total_time if avg_total_time > 0 else 0  # requests per second
            }
            
            # Generate recommendations
            self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> None:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        avg_total_time = analysis['performance_metrics']['avg_total_time']
        avg_processing_time = analysis['performance_metrics']['avg_processing_time']
        success_rate = analysis['success_rate']
        
        # Performance thresholds
        if avg_total_time > 10.0:
            recommendations.append("CRITICAL: Average total time is over 10 seconds - system is too slow for production use")
        elif avg_total_time > 5.0:
            recommendations.append("WARNING: Average total time is over 5 seconds - consider optimization")
        elif avg_total_time < 2.0:
            recommendations.append("GOOD: Average total time is under 2 seconds - performance is acceptable")
        
        if avg_processing_time > 8.0:
            recommendations.append("CRITICAL: Average processing time is over 8 seconds - video processing is too slow")
        elif avg_processing_time > 4.0:
            recommendations.append("WARNING: Average processing time is over 4 seconds - consider optimizing video processing")
        
        if success_rate < 0.8:
            recommendations.append("CRITICAL: Success rate is below 80% - investigate failures")
        elif success_rate < 0.95:
            recommendations.append("WARNING: Success rate is below 95% - monitor for issues")
        else:
            recommendations.append("GOOD: Success rate is above 95% - system is reliable")
        
        # Throughput analysis
        throughput = analysis['performance_metrics']['throughput']
        if throughput < 0.1:
            recommendations.append("CRITICAL: Throughput is below 0.1 requests/second - system cannot handle load")
        elif throughput < 0.2:
            recommendations.append("WARNING: Throughput is below 0.2 requests/second - consider scaling")
        else:
            recommendations.append("GOOD: Throughput is acceptable for current load")
        
        analysis['recommendations'] = recommendations
    
    def print_analysis(self, analysis: Dict) -> None:
        """Print formatted analysis results."""
        print("=" * 80)
        print("VIDEO LIVENESS PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print(f"Total Sessions: {analysis['total_sessions']}")
        print(f"Completed Sessions: {analysis['completed_sessions']}")
        print(f"Failed Sessions: {analysis['failed_sessions']}")
        print(f"Success Rate: {analysis['success_rate']:.2%}")
        print()
        
        if analysis['timing_analysis']:
            print("TIMING ANALYSIS:")
            print("-" * 40)
            
            for metric, stats in analysis['timing_analysis'].items():
                print(f"{metric.replace('_', ' ').title()}:")
                for stat, value in stats.items():
                    print(f"  {stat.upper()}: {value:.3f}s")
                print()
        
        if analysis['performance_metrics']:
            print("PERFORMANCE METRICS:")
            print("-" * 40)
            for metric, value in analysis['performance_metrics'].items():
                if 'time' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.3f}s")
                elif 'efficiency' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
                elif 'throughput' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.3f} req/s")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value}")
            print()
        
        if analysis['recommendations']:
            print("RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"{i}. {rec}")
            print()
        
        # Overall assessment
        print("OVERALL ASSESSMENT:")
        print("-" * 40)
        avg_total_time = analysis['performance_metrics'].get('avg_total_time', 0)
        success_rate = analysis['success_rate']
        
        if avg_total_time < 3.0 and success_rate > 0.95:
            print("✅ SYSTEM PERFORMANCE IS EXCELLENT")
            print("   The video liveness system is performing well within acceptable parameters.")
        elif avg_total_time < 5.0 and success_rate > 0.9:
            print("✅ SYSTEM PERFORMANCE IS GOOD")
            print("   The video liveness system is performing adequately for production use.")
        elif avg_total_time < 8.0 and success_rate > 0.8:
            print("⚠️  SYSTEM PERFORMANCE NEEDS ATTENTION")
            print("   The video liveness system is functional but could benefit from optimization.")
        else:
            print("❌ SYSTEM PERFORMANCE IS UNACCEPTABLE")
            print("   The video liveness system requires immediate attention and optimization.")
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze video liveness performance logs')
    parser.add_argument('--log-file', default='/app/logs/performance.log', 
                       help='Path to performance log file')
    parser.add_argument('--output', help='Output results to JSON file')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.log_file)
    analyzer.parse_log_file()
    analysis = analyzer.analyze_performance()
    
    analyzer.print_analysis(analysis)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to {args.output}")

if __name__ == "__main__":
    main()
