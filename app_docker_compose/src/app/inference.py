"""
Inference functions
"""

def register_face(model_name: str,
                  inference_mode: str,
                  file_path: str,
                  **kwargs):
    return f"{model_name}_{inference_mode}_{file_path}"


def recognize_face(model_name: str,
                   inference_mode: str,
                   file_path: str,
                   threshold: float,
                   **kwargs):
    return f"{model_name}_{inference_mode}_{file_path}_{threshold}"
