# ur-voice-control

### This repository implements the software to operate a Universal Robot arm using voice control. This repo features
- A bespoke neural network to convert voice signals to words
- An agentic AI based control method that responds to the voice commands and articulates a connected UR robotic arm

### User Guide

#### Prerequisites 
- The flexible voice patch
- A Universal Robotic's robot arm 
- OpenAI API key set an environment variable
- Google API Key and Search Engine ID (for ./backend/tools/drawing_tool.py)

To run, ensure the voice patch is attached to the user and powered on and that the host device is connected to robot arm via ethernet. Then open two terminal instances

#### To run the frontend, that provides an interactive interface to view the agent's response, run: 
- cd frontend 
- npm install
- npm run dev

#### To run the backend, that launches the agent, run: 
- cd backend
- pip -i requirements.txt
- python server.py
