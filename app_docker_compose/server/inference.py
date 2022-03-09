

def run_inference(model_name: str,
                  inference_mode: str,
                  file_path: str,
                  threshold: float):
    return f"{model_name}_{inference_mode}_{file_path}_{threshold}"
