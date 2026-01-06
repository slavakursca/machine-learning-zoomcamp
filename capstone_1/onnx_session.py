import os
import sys
import onnxruntime as ort

MODEL_PATH = "train_model/room_classifier_final.onnx"

class ModelSession:
    """Singleton class to manage ONNX model session"""
    
    _instance = None
    _session = None
    _input_name = None
    _output_name = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSession, cls).__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Initialize the ONNX session (called once)"""
        if self._initialized:
            return
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
            print(f"Current directory: {os.getcwd()}", file=sys.stderr)
            print(f"Files in current directory: {os.listdir('.')}", file=sys.stderr)
            if os.path.exists('train_model'):
                print(f"Files in train_model: {os.listdir('train_model')}", file=sys.stderr)
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Get available providers
        available_providers = ort.get_available_providers()
        
        # Use CoreML if available (macOS), otherwise CPU
        providers = []
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Silence warnings
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Error only
        
        print("Initializing ONNX model session...", file=sys.stderr)
        try:
            self._session = ort.InferenceSession(
                MODEL_PATH,
                sess_options=session_options,
                providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._initialized = True
            print(f"Model session initialized successfully!", file=sys.stderr)
            print(f"Using providers: {self._session.get_providers()}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR initializing model session: {e}", file=sys.stderr)
            raise
    
    @property
    def session(self):
        """Get the ONNX session"""
        if not self._initialized:
            self.initialize()
        return self._session
    
    @property
    def input_name(self):
        """Get the input tensor name"""
        if not self._initialized:
            self.initialize()
        return self._input_name
    
    @property
    def output_name(self):
        """Get the output tensor name"""
        if not self._initialized:
            self.initialize()
        return self._output_name