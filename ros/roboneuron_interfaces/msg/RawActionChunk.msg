# RawActionChunk.msg

# Semantic protocol for the flattened actions, e.g. "normalized_cartesian_velocity".
string protocol

# Reference frame for Cartesian actions, e.g. "tool" or "base".
string frame

# Per-step action dimensionality.
uint32 action_dim

# Number of steps encoded in `values`.
uint32 chunk_length

# Suggested spacing between chunk steps.
float64 step_duration_sec

# Flattened action values with shape (chunk_length, action_dim).
float64[] values
