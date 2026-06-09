# PersonaAct

**PersonaAct** is a framework for generating personalized LLM-based agents from real user behavioral data to audit filter bubbles in short-video recommendation systems.

## Overview

PersonaAct consists of three components:
1. **Multimodal Dataset**: Short-video user actions with video frames, audio transcripts, and interaction sequences
2. **Interview Agent**: Synthesizes user personas through behavioral analysis and targeted questioning
3. **Persona-driven Agents**: Deployed to audit filter bubble formation in recommendation systems

## Installation

```bash
git clone https://github.com/Silung/PersonaAct.git
cd PersonaAct

# Install dependencies
pip install -r requeriment.txt
cd interview && pip install -r requirements.txt && cd ..
```

## Quick Start

### Data Preparation
```bash
python prepare_data.py --input_dir raw_data --output_dir data
```

### Run Interview Agent
```bash
cd interview
python app.py
```

### Inference
```bash
python infer.py --model_path <path_to_model> --data_path data/
```

## Dataset

The dataset contains:
- 4,485 video interaction samples from 8 personas across 86 sessions
- Video frames (1 FPS), audio transcripts, and user actions
- 25+ content categories with metadata
