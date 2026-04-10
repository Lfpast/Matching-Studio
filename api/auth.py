from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, timedelta, timezone
import yaml
import logging
from jose import jwt, ExpiredSignatureError, JWTError
from fastapi import HTTPException

@dataclass
class Credential:
    """认证凭证数据类"""
    username: str
    password: str

@dataclass
class TokenPayload:
    """JWT Token载荷"""
    username: str
    exp: int  # 过期时间戳
    iat: int  # 签发时间戳

# JWT配置
ALGORITHM = "HS256"

def load_credentials_from_config(config_path: str) -> Dict[str, str]:
    """从config.yaml读取认证凭证"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        credentials_list = config.get('auth', {}).get('credentials', [])
        credentials_dict = {}
        for cred in credentials_list:
            credentials_dict[cred['username']] = cred['password']
        return credentials_dict
    except Exception as e:
        logging.warning(f"Failed to load credentials from {config_path}: {e}")
        return {}

def verify_password(username: str, password: str, credentials_dict: Dict[str, str]) -> bool:
    """验证用户名和密码是否正确"""
    if username in credentials_dict:
        return credentials_dict[username] == password
    return False

def create_token(username: str, config_dict: dict, expires_delta: Optional[timedelta] = None) -> str:
    """为用户生成JWT token"""
    auth_config = config_dict.get("auth", {})
    jwt_secret = auth_config.get("jwt_secret", "your_secret_key_here")
    
    if expires_delta is None:
        ttl_minutes = auth_config.get("token_ttl_minutes", 60)
        expires_delta = timedelta(minutes=ttl_minutes)
        
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    to_encode = {
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    
    encoded_jwt = jwt.encode(to_encode, jwt_secret, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, config_dict: dict) -> str:
    """验证JWT token并提取用户名"""
    auth_config = config_dict.get("auth", {})
    jwt_secret = auth_config.get("jwt_secret", "your_secret_key_here")
    
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def extract_token_from_header(authorization_header: str) -> str:
    """从HTTP Authorization header提取Bearer token"""
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return authorization_header.split(" ")[1]
