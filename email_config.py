"""
Email Configuration for IoT Anomaly Detection System
"""

import os
from typing import Optional

def setup_email_config():
    """
    Setup email configuration with user input
    """
    print("📧 Email Alert Configuration")
    print("=" * 40)
    
    # Check if environment variables are already set
    email_from = os.getenv("EMAIL_FROM")
    email_password = os.getenv("EMAIL_PASSWORD")
    
    if email_from and email_password:
        print(f"✅ Email already configured: {email_from}")
        return True
    
    print("To enable email alerts, you need:")
    print("1. A Gmail account")
    print("2. 2-factor authentication enabled")
    print("3. An app password generated")
    print()
    
    # Get user input
    email = input("Enter your Gmail address (or press Enter to skip): ").strip()
    if not email:
        print("⚠️  Email alerts will be disabled")
        return False
    
    password = input("Enter your app password: ").strip()
    if not password:
        print("⚠️  Email alerts will be disabled")
        return False
    
    # Set environment variables
    os.environ["EMAIL_FROM"] = email
    os.environ["EMAIL_PASSWORD"] = password
    
    print(f"✅ Email configured: {email}")
    print("📝 Note: These settings are for this session only.")
    print("   To make them permanent, add them to your system environment variables.")
    
    return True

def get_email_config() -> tuple[Optional[str], Optional[str]]:
    """
    Get email configuration from environment variables
    """
    email_from = os.getenv("EMAIL_FROM")
    email_password = os.getenv("EMAIL_PASSWORD")
    return email_from, email_password

def is_email_configured() -> bool:
    """
    Check if email is properly configured
    """
    email_from, email_password = get_email_config()
    return email_from is not None and email_password is not None

if __name__ == "__main__":
    setup_email_config()
