import sys
print("Python version:", sys.version)

try:
    import PIL
    print("PIL version:", PIL.__version__)
except Exception as e:
    print("PIL error:", e)

try:
    import transformers
    print("transformers version:", transformers.__version__)
except Exception as e:
    print("transformers error:", e)

try:
    import torch
    print("torch version:", torch.__version__)
except Exception as e:
    print("torch error:", e)
