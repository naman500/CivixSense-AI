# Alert Manager for Crowd Detection System
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

class AlertManager:
    def __init__(self):
        """Initialize alert management system with configuration and history"""
        self.config_file = os.path.join(os.path.dirname(__file__), 'alert_config.json')
        self.alert_history_file = os.path.join(os.path.dirname(__file__), 'alert_history.json')
        self.config: Dict[str, Any] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.load_config()
        self.alert_history = self.load_alert_history()
        
    def load_config(self) -> None:
        """Load alert configuration from file with error handling"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self._create_default_config()
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create and save default configuration"""
        self.config = {
            "cooldown_period": 60,  # seconds between alerts for the same camera
            "alert_actions": {
                "low": ["Notify security personnel for manual crowd management"],
                "medium": ["Close certain entrances", "Redirect people to less crowded areas"],
                "high": ["Open additional exits", "Implement emergency crowd control measures"]
            }
        }
        self.save_config()
    
    def save_config(self) -> None:
        """Save alert configuration to file with error handling"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            
    def load_alert_history(self) -> List[Dict[str, Any]]:
        """Load alert history from file with error handling"""
        if not os.path.exists(self.alert_history_file):
            return []
            
        try:
            with open(self.alert_history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading alert history: {str(e)}")
            return []
    
    def save_alert_history(self) -> None:
        """Save alert history to file with error handling"""
        try:
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=4)
        except Exception as e:
            print(f"Error saving alert history: {str(e)}")

    def check_threshold(self, camera_id: str, crowd_count: int, threshold: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if crowd count exceeds threshold and generate alert if needed
        
        Parameters:
            camera_id: ID of the camera
            crowd_count: Current crowd count
            threshold: Threshold value for alerts
            
        Returns:
            Tuple of (alert_generated, alert_info)
        """
        # No alert if count is below threshold
        if crowd_count < threshold:
            self.clear_alert(camera_id)
            return False, None
        
        # Check cooldown period
        current_time = time.time()
        if camera_id in self.active_alerts:
            last_alert_time = self.active_alerts[camera_id]['timestamp']
            if current_time - last_alert_time < self.config['cooldown_period']:
                return False, self.active_alerts[camera_id]
        
        # Determine severity and create alert
        severity = self._determine_severity(crowd_count, threshold)
        actions = self.config['alert_actions'][severity]
        
        alert_info = self._create_alert_info(
            camera_id=camera_id,
            crowd_count=crowd_count,
            threshold=threshold,
            severity=severity,
            actions=actions
        )
        
        # Update active alerts and history
        self.active_alerts[camera_id] = alert_info
        self._update_alert_history(alert_info)
        
        return True, alert_info
    
    def _determine_severity(self, crowd_count: int, threshold: int) -> str:
        """Determine alert severity based on crowd count ratio"""
        severity_ratio = crowd_count / threshold
        if severity_ratio < 1.2:
            return "low"
        elif severity_ratio < 1.5:
            return "medium"
        return "high"
    
    def _create_alert_info(self, camera_id: str, crowd_count: int, threshold: int, 
                          severity: str, actions: List[str]) -> Dict[str, Any]:
        """Create alert information dictionary"""
        return {
            'camera_id': camera_id,
            'timestamp': time.time(),
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'crowd_count': crowd_count,
            'threshold': threshold,
            'severity': severity,
            'actions': actions
        }
    
    def _update_alert_history(self, alert_info: Dict[str, Any]) -> None:
        """Update alert history with new alert"""
        self.alert_history.append(alert_info)
        if len(self.alert_history) > 100:  # Keep history limited
            self.alert_history = self.alert_history[-100:]
        self.save_alert_history()
    
    def get_suggested_actions(self, camera_id: str) -> List[str]:
        """Get suggested actions for a camera with active alert"""
        return self.active_alerts.get(camera_id, {}).get('actions', [])
    
    def get_alert_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get alert history with optional limit"""
        return self.alert_history[-limit:]
    
    def clear_alert(self, camera_id: str) -> bool:
        """Clear active alert for a camera"""
        if camera_id in self.active_alerts:
            del self.active_alerts[camera_id]
            return True
        return False

    def remove_camera_alerts(self, camera_id: str) -> bool:
        """Remove all alerts associated with a specific camera"""
        try:
            # First clear active alert
            self.clear_alert(camera_id)
            
            # Remove from alert history
            self.alert_history = [alert for alert in self.alert_history if alert.get('camera_id') != camera_id]
            
            # Save updated history
            self.save_alert_history()
            
            return True
        except Exception as e:
            print(f"Error removing alerts for camera {camera_id}: {str(e)}")
            return False

    def add_alert(self, camera_id: str, camera_name: str, crowd_count: int, 
                 threshold: int, severity: str, actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add a new alert to the system"""
        alert_info = self._create_alert_info(
            camera_id=camera_id,
            crowd_count=crowd_count,
            threshold=threshold,
            severity=severity,
            actions=actions or self.config['alert_actions'].get(severity, [])
        )
        
        # Update active alerts and history
        self.active_alerts[camera_id] = alert_info
        self._update_alert_history(alert_info)
        
        return alert_info

    def reset_alert_history(self) -> bool:
        """Reset alert history and save empty history file"""
        try:
            self.alert_history = []
            self.active_alerts = {}
            self.save_alert_history()
            return True
        except Exception as e:
            print(f"Error resetting alert history: {str(e)}")
            return False
