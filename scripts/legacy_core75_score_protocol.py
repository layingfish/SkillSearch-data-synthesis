"""Shared constants for legacy Official core75 score diagnostics."""

COMPONENT_LABELS = (
    "code_position_0",
    "code_position_1",
    "code_position_2",
    "code_position_3",
    "stop_control_1",
)
CUTOFFS = (1, 5, 10, 50)
DEEP_GATES = {"@10": 0.69, "@50": 0.75}
RTX_R1_REFERENCE = 0.28368253968253965
R1_GATE = RTX_R1_REFERENCE - 0.01
