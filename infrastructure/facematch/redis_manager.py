"""
Redis Session Manager for Liveness Challenges
Manages session data for eye tracking liveness challenges with Redis caching.
"""

import json
import logging
import random
import time
import uuid
from typing import Dict, List, Optional, Any
import redis
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LivenessSession:
    """Represents a liveness challenge session."""
    session_id: str
    sequence: List[int]
    created_at: float
    expires_at: float
    area_duration: float = 3.0
    status: str = "active"  # active, completed, expired
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    screen_areas: Optional[List[Dict]] = None
    face_snapshot_size: Optional[int] = None

class RedisSessionManager:
    """Manages liveness challenge sessions and results using Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 session_ttl: int = 300, 
                 result_ttl: int = 3600):  # 5 minutes for sessions, 1 hour for results
        """
        Initialize Redis session manager.
        
        Args:
            redis_url: Redis connection URL
            session_ttl: Session time-to-live in seconds
            result_ttl: Result time-to-live in seconds (1 hour default)
        """
        self.session_ttl = session_ttl
        self.result_ttl = result_ttl
        self.redis_client = None
        self._connect_to_redis(redis_url)
    
    def _connect_to_redis(self, redis_url: str) -> None:
        """
        Connect to Redis server.
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Redis not available - sessions will not be persistent")
            self.redis_client = None
    
    def generate_sequence(self, count: int = 3) -> List[int]:
        """
        Generate random sequence of screen areas (1-6).
        
        Args:
            count: Number of areas to select
            
        Returns:
            List of random area numbers
        """
        if count > 6:
            count = 6
        
        # Select random areas without repetition
        areas = list(range(1, 7))  # Areas 1-6
        return random.sample(areas, count)
    
    def create_session(self, area_duration: float = 3.0) -> LivenessSession:
        """
        Create a new liveness challenge session.
        
        Args:
            area_duration: Duration for each area in seconds
            
        Returns:
            LivenessSession object
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        
        session = LivenessSession(
            session_id=session_id,
            sequence=[],  # Will be generated later with screen data
            created_at=now,
            expires_at=now + self.session_ttl,
            area_duration=area_duration,
            status="active"
        )
        
        # Store in Redis if available
        if self.redis_client:
            try:
                session_data = asdict(session)
                self.redis_client.setex(
                    f"liveness_session:{session_id}",
                    self.session_ttl,
                    json.dumps(session_data)
                )
                logger.info(f"Created liveness session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to store session in Redis: {e}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[LivenessSession]:
        """
        Retrieve a liveness challenge session.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            LivenessSession object or None if not found
        """
        if not self.redis_client:
            logger.warning("Redis not available - cannot retrieve session")
            return None
        
        try:
            session_data = self.redis_client.get(f"liveness_session:{session_id}")
            if not session_data:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            data = json.loads(session_data)
            session = LivenessSession(**data)
            
            # Check if session has expired
            if time.time() > session.expires_at:
                logger.warning(f"Session expired: {session_id}")
                self.delete_session(session_id)
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to retrieve session: {e}")
            return None
    
    def update_session_data(self, session_id: str, data_update: Dict[str, Any]) -> bool:
        """
        Update session data with new information.
        
        Args:
            session_id: Session ID to update
            data_update: Dictionary of data to update
            
        Returns:
            True if updated successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Update session attributes
        for key, value in data_update.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        try:
            session_data = asdict(session)
            self.redis_client.setex(
                f"liveness_session:{session_id}",
                self.session_ttl,
                json.dumps(session_data)
            )
            logger.info(f"Updated session data: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update session data: {e}")
            return False
    
    def update_session_status(self, session_id: str, status: str) -> bool:
        """
        Update the status of a session.
        
        Args:
            session_id: Session ID to update
            status: New status (active, completed, expired)
            
        Returns:
            True if updated successfully
        """
        return self.update_session_data(session_id, {'status': status})
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Redis.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(f"liveness_session:{session_id}")
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def save_liveness_result(self, session_id: str, result_data: Dict[str, Any], 
                           third_party_id: Optional[str] = None) -> bool:
        """
        Save liveness challenge result to Redis with 1-hour expiration.
        
        Args:
            session_id: Session ID
            result_data: Complete result data from liveness validation
            third_party_id: Optional 3rd party identifier
            
        Returns:
            True if saved successfully
        """
        if not self.redis_client:
            return False
        
        try:
            # Create result key with optional 3rd party ID
            if third_party_id:
                result_key = f"liveness_result:{third_party_id}:{session_id}"
            else:
                result_key = f"liveness_result:{session_id}"
            
            # Add metadata to result
            result_with_metadata = {
                **result_data,
                'stored_at': time.time(),
                'session_id': session_id,
                'third_party_id': third_party_id,
                'expires_at': time.time() + self.result_ttl
            }
            
            # Store in Redis with 1-hour expiration
            self.redis_client.setex(
                result_key,
                self.result_ttl,
                json.dumps(result_with_metadata)
            )
            
            logger.info(f"Saved liveness result: {result_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save liveness result: {e}")
            return False
    
    def get_liveness_result(self, session_id: str, 
                          third_party_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve liveness challenge result from Redis.
        
        Args:
            session_id: Session ID
            third_party_id: Optional 3rd party identifier
            
        Returns:
            Result data dictionary or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            # Create result key with optional 3rd party ID
            if third_party_id:
                result_key = f"liveness_result:{third_party_id}:{session_id}"
            else:
                result_key = f"liveness_result:{session_id}"
            
            result_json = self.redis_client.get(result_key)
            if result_json:
                result_data = json.loads(result_json.decode('utf-8'))
                logger.info(f"Retrieved liveness result: {result_key}")
                return result_data
            else:
                logger.info(f"No result found for: {result_key}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve liveness result: {e}")
            return None
    
    def delete_liveness_result(self, session_id: str, 
                             third_party_id: Optional[str] = None) -> bool:
        """
        Delete a liveness result from Redis.
        
        Args:
            session_id: Session ID
            third_party_id: Optional 3rd party identifier
            
        Returns:
            True if deleted successfully
        """
        if not self.redis_client:
            return False
        
        try:
            # Create result key with optional 3rd party ID
            if third_party_id:
                result_key = f"liveness_result:{third_party_id}:{session_id}"
            else:
                result_key = f"liveness_result:{session_id}"
            
            self.redis_client.delete(result_key)
            logger.info(f"Deleted liveness result: {result_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete liveness result: {e}")
            return False
    
    def clean_expired_sessions(self) -> int:
        """
        Clean up expired sessions from Redis.
        
        Returns:
            Number of sessions cleaned
        """
        if not self.redis_client:
            return 0
        
        try:
            # Get all session keys
            keys = self.redis_client.keys("liveness_session:*")
            cleaned_count = 0
            
            for key in keys:
                session_data = self.redis_client.get(key)
                if session_data:
                    try:
                        data = json.loads(session_data)
                        if time.time() > data.get('expires_at', 0):
                            self.redis_client.delete(key)
                            cleaned_count += 1
                    except json.JSONDecodeError:
                        # Invalid data, delete it
                        self.redis_client.delete(key)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to clean expired sessions: {e}")
            return 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions.
        
        Returns:
            Dictionary containing session statistics
        """
        if not self.redis_client:
            return {
                'redis_available': False,
                'total_sessions': 0,
                'active_sessions': 0,
                'expired_sessions': 0
            }
        
        try:
            keys = self.redis_client.keys("liveness_session:*")
            total_sessions = len(keys)
            active_sessions = 0
            expired_sessions = 0
            
            current_time = time.time()
            
            for key in keys:
                session_data = self.redis_client.get(key)
                if session_data:
                    try:
                        data = json.loads(session_data)
                        if current_time <= data.get('expires_at', 0):
                            active_sessions += 1
                        else:
                            expired_sessions += 1
                    except json.JSONDecodeError:
                        expired_sessions += 1
            
            return {
                'redis_available': True,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'expired_sessions': expired_sessions,
                'session_ttl': self.session_ttl
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                'redis_available': False,
                'error': str(e)
            }

# Global session manager instance
session_manager = None

def initialize_session_manager(redis_url: str = "redis://localhost:6379", 
                             session_ttl: int = 300) -> RedisSessionManager:
    """
    Initialize global session manager.
    
    Args:
        redis_url: Redis connection URL
        session_ttl: Session time-to-live in seconds
        
    Returns:
        RedisSessionManager instance
    """
    global session_manager
    session_manager = RedisSessionManager(redis_url, session_ttl)
    return session_manager

def get_session_manager() -> Optional[RedisSessionManager]:
    """
    Get the global session manager instance.
    
    Returns:
        RedisSessionManager instance or None
    """
    global session_manager
    return session_manager 