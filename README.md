# Wearable, Broadband Auscultation Patch with Cantilever Pressure Transducer for Remote Healthcare Monitoring
## Overview
The **AusculPatch** is a wireless, flexible auscultation wearable that overcomes the frequency limitations of standard MEMS sensors by utilizing a highly sensitive, nanothin **cantilever pressure transducer (CPT)**. This innovation enables precise monitoring across a vast acoustic range (**0.2Hz to 10kHz**), capturing diverse signals such as pulse waves, cardiac sounds, and respiration. Weighing just **3.2g** with a low **4.5mW** power draw, its compact single-chip design allows for comfortable, continuous wear. Ultimately, the AusculPatch provides a versatile platform for AI-assisted diagnostics, sleep assessment, and human-machine interaction, offering a lightweight alternative to traditional medical stethoscopes for home-based healthcare.

![](https://github.com/bach-dt/2026_AusculPatch/blob/main/overview.png)

## ur-voice-control

### This repository implements the software to operate a Universal Robot arm using voice control. This repo features
- A bespoke neural network to convert voice signals to words
- An agentic AI based control method that responds to the voice commands and articulates a connected UR robotic arm

### User Guide

#### Prerequisites 
- The AusculPatch
- A Universal Robotic's robot arm 
- OpenAI API key set an environment variable
- Google API Key and Search Engine ID (for ./backend/tools/drawing_tool.py)

To run, ensure the AusculPatch is attached to the user and that the host device is connected to robot arm via ethernet. Then open two terminal instances.

#### Clone respository
Clone this repository and navigate to the root directory.
```
git clone https://github.com/bach-dt/2026_AusculPatch.git
cd ur-voice-control-final
```

#### To run the frontend, that provides an interactive interface to view the agent's response, run: 
```
cd frontend 
npm install
npm run dev
```

#### To run the backend, that launches the agent, run: 
```
cd backend
pip -i requirements.txt
python server.py
```

## Authors
- [Tran Bach Dang](https://github.com/bach-dt)
- Nicolas Tong
- Contact: z5561902@ad.unsw.edu.au


