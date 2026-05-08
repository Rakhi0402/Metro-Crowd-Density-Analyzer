"""
notifier.py
-----------
Sends notifications through multiple channels:
- Console output
- Email alerts
- SMS alerts (requires Twilio)
- Slack messages
- Local logs
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class Notifier:
    """
    Multi-channel notification system for crowd alerts.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize notifier with optional config file.
        
        Parameters
        ----------
        config_file : Path, optional
            Path to config JSON with API credentials.
        """
        self.config = self._load_config(config_file) if config_file else {}
        self.notification_log = []
        
    def _load_config(self, config_file: Path) -> dict:
        """Load credentials from config file."""
        try:
            with open(config_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def notify_console(self, message: str, level: str = "INFO"):
        """Send notification to console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_icons = {
            "INFO": "ℹ️ ",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "CRITICAL": "🚨",
            "SUCCESS": "✅"
        }
        icon = level_icons.get(level.upper(), "•")
        print(f"{icon} [{timestamp}] {message}")
    
    def notify_email(self, recipient: str, subject: str, body: str) -> bool:
        """
        Send email notification (requires SMTP config).
        
        Parameters
        ----------
        recipient : str
            Email address.
        subject : str
            Email subject.
        body : str
            Email body.
        
        Returns
        -------
        bool
            Success status.
        """
        if 'smtp' not in self.config:
            print("⚠️  Email not configured. Add SMTP settings to config.")
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            cfg = self.config['smtp']
            msg = MIMEMultipart()
            msg['From'] = cfg['sender']
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(cfg['server'], cfg['port']) as server:
                server.starttls()
                server.login(cfg['username'], cfg['password'])
                server.send_message(msg)
            
            self.notification_log.append({
                'type': 'email',
                'recipient': recipient,
                'timestamp': datetime.now(),
                'success': True
            })
            self.notify_console(f"✉️  Email sent to {recipient}", "SUCCESS")
            return True
        except Exception as e:
            self.notify_console(f"❌ Email failed: {e}", "ERROR")
            return False
    
    def notify_sms(self, phone: str, message: str) -> bool:
        """
        Send SMS alert (requires Twilio account).
        
        Parameters
        ----------
        phone : str
            Phone number with country code.
        message : str
            Message text.
        
        Returns
        -------
        bool
            Success status.
        """
        if 'twilio' not in self.config:
            print("⚠️  SMS not configured. Add Twilio settings to config.")
            return False
        
        try:
            from twilio.rest import Client # pyright: ignore[reportMissingImports]
            cfg = self.config['twilio']
            client = Client(cfg['account_sid'], cfg['auth_token'])
            
            message = client.messages.create(
                body=message,
                from_=cfg['from_number'],
                to=phone
            )
            
            self.notification_log.append({
                'type': 'sms',
                'recipient': phone,
                'timestamp': datetime.now(),
                'success': True
            })
            self.notify_console(f"📱 SMS sent to {phone}", "SUCCESS")
            return True
        except Exception as e:
            self.notify_console(f"❌ SMS failed: {e}", "ERROR")
            return False
    
    def notify_slack(self, webhook_url: str, message: str, channel: str = None) -> bool:
        """
        Send Slack message (requires webhook URL).
        
        Parameters
        ----------
        webhook_url : str
            Slack incoming webhook URL.
        message : str
            Message text.
        channel : str, optional
            Slack channel override.
        
        Returns
        -------
        bool
            Success status.
        """
        try:
            import requests
            
            payload = {'text': message}
            if channel:
                payload['channel'] = channel
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                self.notification_log.append({
                    'type': 'slack',
                    'channel': channel or 'default',
                    'timestamp': datetime.now(),
                    'success': True
                })
                self.notify_console(f"💬 Slack message sent", "SUCCESS")
                return True
            else:
                raise Exception(f"Slack API error: {response.status_code}")
        except Exception as e:
            self.notify_console(f"❌ Slack failed: {e}", "ERROR")
            return False
    
    def alert_overcrowding(self, coach_id: int, crowd_count: int, 
                          threshold: int, channels: list[str] = None):
        """
        Send overcrowding alert through specified channels.
        
        Parameters
        ----------
        coach_id : int
            Coach/coach identifier.
        crowd_count : int
            Current crowd density.
        threshold : int
            Alert threshold.
        channels : list[str]
            Channels to use: ['console', 'email', 'sms', 'slack'].
        """
        channels = channels or ['console']
        
        # Console alert
        if 'console' in channels:
            self.notify_console(
                f"🚨 OVERCROWDING ALERT: Coach {coach_id} - {crowd_count} people "
                f"(threshold: {threshold})",
                "CRITICAL"
            )
        
        # Email alert (if configured)
        if 'email' in channels and 'email_recipient' in self.config:
            subject = f"🚨 Metro Train Alert: Overcrowding in Coach {coach_id}"
            body = f"""
            <h2>Overcrowding Alert</h2>
            <p><strong>Coach ID:</strong> {coach_id}</p>
            <p><strong>Current Density:</strong> {crowd_count} people</p>
            <p><strong>Threshold:</strong> {threshold} people</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Action:</strong> Limit entry, deploy staff, consider rerouting.</p>
            """
            self.notify_email(self.config['email_recipient'], subject, body)
        
        # SMS alert (if configured)
        if 'sms' in channels and 'sms_recipient' in self.config:
            msg = (f"ALERT: Coach {coach_id} overcrowded with {crowd_count} people. "
                   f"Threshold: {threshold}. Take action now.")
            self.notify_sms(self.config['sms_recipient'], msg)
    
    def save_notification_log(self, filepath: Path):
        """Save notification history to file."""
        import csv
        
        if not self.notification_log:
            return
        
        with open(filepath, 'w', newline='') as f:
            if self.notification_log:
                writer = csv.DictWriter(f, fieldnames=self.notification_log[0].keys())
                writer.writeheader()
                for entry in self.notification_log:
                    writer.writerow(entry)
        
        print(f"✓ Notification log saved: {filepath}")


def create_example_config(filepath: Path):
    """Create example configuration file."""
    config = {
        "email_recipient": "operator@metro.com",
        "sms_recipient": "+1234567890",
        "smtp": {
            "server": "smtp.gmail.com",
            "port": 587,
            "sender": "alerts@metro.com",
            "username": "your_email@gmail.com",
            "password": "your_app_password"
        },
        "twilio": {
            "account_sid": "your_account_sid",
            "auth_token": "your_auth_token",
            "from_number": "+1234567890"
        },
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Example config created: {filepath}")
    print("  Edit this file and add your credentials to enable notifications.")
