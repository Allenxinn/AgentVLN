from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NavDatasetConfig:
    """Configuration for VLN Navigation Dataset.
    
    Controls prompt template, output prefixes, action display,
    distance thresholds, and history settings.
    """
    prompt_template: str = (
        "You are an autonomous navigation assistant. "
        "Your task is to <instruction>. "
        "Where should you go next to stay on track? "
        "Select the most suitable waypoint coordinate from the following <frontiers_coord>. "
        "If no suitable coordinate is found, output the action command. "
        "If the destination is visible, output the coordinates of the target point directly. "
        "Please output STOP when you have successfully completed the task. "
        "These are your historical observations: <history>. "
        "<conjunction><image>."
    )
    
    prompt_template_no_frontiers: str = (
        "You are an autonomous navigation assistant. "
        "Your task is to <instruction>. "
        "Where should you go next to stay on track? "
        "Enter the action command directly. "
        "If you can see the endpoint, enter the coordinates of the target point. "
        "Please output STOP when you have successfully completed the task. "
        "These are your historical observations: <history>. "
        "<conjunction><image>."
    )
    
    conjunctions: List[str] = field(default_factory=lambda: [
        "Now observe: ",
        "Current view: ",
        "Looking ahead: ",
        "What you see now: ",
        "Your current observation: ",
    ])
    
    frontier_prefix: str = "<frontiers_coord>"  
    target_prefix: str = "<target>"             
    action_prefix: str = "<action>"            
    
    prefix_trainable: bool = True
    
    action_names: Dict[int, str] = field(default_factory=lambda: {
        0: "STOP",
        1: "FORWARD",
        2: "TURN_LEFT",
        3: "TURN_RIGHT",
    })
    
    action_symbols: Dict[int, str] = field(default_factory=lambda: {
        0: "⏹",
        1: "↑",
        2: "←",
        3: "→",
    })
    
    use_action_symbols: bool = False
    distance_threshold: float = 2.0
    pixel_distance_threshold: float = 100.0
    max_history_frames: int = 8
    max_action_lookahead: int = 3
    image_size: Tuple[int, int] = (480, 640)
    image_token: str = "<image>"
    image_ext: str = "jpg"

    start_token: str = "<|im_start|>"
    stop_token: str = "<|im_end|>"
    

    coord_format: str = "({u}, {v})"
    
    def get_action_display(self, action_code: int) -> str:
        if self.use_action_symbols:
            return self.action_symbols.get(action_code, f"UNKNOWN({action_code})")
        return self.action_names.get(action_code, f"UNKNOWN({action_code})")
    
    def format_coord(self, u: int, v: int) -> str:
        return self.coord_format.format(u=u, v=v)
    
    def format_coord_list(self, coords: List[List[int]]) -> str:
        formatted = [self.format_coord(c[0], c[1]) for c in coords if c is not None]
        return "[" + ", ".join(formatted) + "]"
