# DocGPT (Doctor GPT) - AI-Powered Medical Diagnosis System

[![Docker Image Size](https://img.shields.io/docker/image-size/codewithdark/docgpt?label=Docker%20Image%20Size&logo=docker)](https://hub.docker.com/r/codewithdark/docgpt)
[![Docker Pulls](https://img.shields.io/docker/pulls/codewithdark/docgpt?logo=docker)](https://hub.docker.com/r/codewithdark/docgpt)


DocGPT (Doctor GPT) is an advanced medical diagnosis system that combines Vision Transformer (ViT) based deep learning models with LangChain agents to provide comprehensive medical image analysis and detailed diagnostic reports. The system leverages the power of PyTorch for deep learning and Groq's LLM for generating human-like medical insights.

## Core Technology

- **LangChain Agents**: Intelligent agents that coordinate between different disease detection models and the LLM to provide comprehensive medical analysis
- **Vision Transformer (ViT)**: State-of-the-art transformer architecture for medical image analysis
- **Deep Learning Models**: Specialized PyTorch models trained for different medical conditions:
  - ResNet-based architecture for Eye Disease detection
  - Vision Transformer for Skin Cancer classification
  - Custom CNN architecture for Pneumonia detection

## Features

- **Multi-Disease Detection**: Supports multiple medical conditions:
  - Eye Diseases (Cataract, Glaucoma, Diabetic Retinopathy)
  - Skin Cancer (Melanoma Detection)
  - Pneumonia (X-ray Analysis)
  - Brain Tumor (Coming Soon)
  - Heart Disease (Coming Soon)
  - [See the DocGPT models Code](https://github.com/XCollab/DocGPT-Models.git)

- **AI-Powered Analysis**: 
  - Deep learning models for accurate disease detection
  - Groq LLM integration for detailed medical reports
  - LangChain agents for orchestrating the analysis pipeline

- **Modern Architecture**:
  - FastAPI backend with automatic OpenAPI documentation
  - Streamlit frontend for testing (React.js interface planned)
  - Modular design for easy extension to new disease types

## Tech Stack

- **Backend Framework**: FastAPI
- **Deep Learning**: 
  - PyTorch
  - torchvision
  - Vision Transformer (ViT)
- **AI Integration**: 
  - LangChain for agent orchestration
  - Groq API for medical report generation
- **Image Processing**: 
  - PIL
  - torchvision transforms
- **Development**: 
  - Python 3.12+
  - pydantic for data validation
  - uvicorn for ASGI server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/codewithdark-git/DocGPT.git
cd DocGPT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=llama-3.2-11b-vision-preview
```

## Usage

1. Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```

2. Start the Streamlit frontend (in a new terminal):
```bash
streamlit run streamlit_app.py
```

3. Access the applications:
- API Documentation: http://localhost:8000/docs
- Streamlit Interface: http://localhost:8501

## API Endpoints

### Health Check
- `GET /api/v1/health`: Check API health status

### Disease Prediction
- `POST /api/v1/predict`: Submit an image for disease prediction
  - Parameters:
    - `disease_type`: Type of disease to predict
    - `file`: Image file


## Project Structure

```
DocGPT/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── routers/          # API routes
│   ├── schemas/          # Pydantic models
│   └── services/         # ML models and business logic
├── models/               # Trained model files
├── streamlit_app.py      # Streamlit frontend
├── requirements.txt      # Project dependencies
└── .env                  # Environment variables
```

## Disease Types

- **Eye Disease Detection**:
  - Normal
  - Cataract
  - Glaucoma
  - Diabetic Retinopathy

- **Skin Cancer Detection**:
  - Melanoma
  - Non-Melanoma

- **Pneumonia Detection**:
  - Normal
  - Pneumonia

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical datasets providers
- PyTorch team
- Groq API team
- FastAPI and Streamlit communities
 
