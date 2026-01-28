import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """
    Represents a profile from an ecommerce system
    """

    id: str = ""
    created_at: str = ""
    cart: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    clickstream: List[Dict[str, Any]] = field(default_factory=list)
    site_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> "UserProfile":
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        # Filter to only include fields defined in the dataclass
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)
   

    @staticmethod
    def tryparse_profile(profile: Union[str, Dict[str, Any]]) -> Optional["UserProfile"]:
        """
        Parse the user profile from a string or dict representation.
        """
        try:
            if isinstance(profile, str):
                return UserProfile.from_json(profile)
            elif isinstance(profile, dict):
                return UserProfile.from_dict(profile)
            else:                
                logger.warning(f"Unsupported profile type: {type(profile)}")
                return None
        except Exception as e:            
            logger.error(f"tryparse_profile Exception: {e}")
            return None