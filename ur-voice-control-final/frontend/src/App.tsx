import { useState } from 'react';
import { FaMicrophone } from "react-icons/fa";
import { FaBluetoothB } from "react-icons/fa";
import { MdRecordVoiceOver } from "react-icons/md";
import Markdown from 'react-markdown'

function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("Results");
  const [voice, setVoice] = useState(false);

  const handleSubmit = async(event : React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      event.stopPropagation();
      const content = event.currentTarget.value
      setInput(content)

      const response = await fetch("http://localhost:5555/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
      });
      
      const data = await response.json();
      parseOutput(data["response"])
    }
  };

  const recognizeSpeech = async () => {
    setInput("Using Voice Recognition")
    const recognition = new ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.start();

    recognition.onresult = async (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);

      const response = await fetch("http://localhost:5555/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: transcript }),
      });

      const data = await response.json();
      parseOutput(data["response"])
    };
  };

  const recognizeBluetooth = async() => {
    setInput("Using Voice Bluetooth")

    const response = await fetch("http://localhost:5555/patch");
    const data = await response.json(); 
    parseOutput(data["response"])
  }

  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      const synth = window.speechSynthesis;
      const utterance = new SpeechSynthesisUtterance(text);

      utterance.lang = 'en-US';
      utterance.rate = 1;
      utterance.pitch = 1;
      synth.speak(utterance);
    } else {
      console.log('Text-to-speech not supported.');
    }
  }

  const parseOutput = (output: string) => {
    setOutput(output);
    if (voice) {
      speakText(output);
    }
  }
  // handle bluetooth

  return (
    <div className="flex h-screen w-screen flex-col justify-center items-center bg-white">
      <div className="font-serif text-6xl font-bold tracking-normal min-w-128">
        <span className='text-red-400 hover:text-red-500 transition-all cursor-pointer'>UR</span> Voice Control Agent
      </div>
      <div className="w-1/2 flex gap-2 mt-8 min-w-128">
        <input 
          placeholder="What do you want to do?" 
          className="w-full rounded-xl px-4 py-2 shadow-md font-mono tracking-tight focus:outline-none focus:inset-ring-1 focus:inset-ring-gray-400" 
          onKeyDown={handleSubmit}
        />
        <button 
          className="rounded-xl px-4 bg-red-400 shadow-md cursor-pointer pointer-events-auto hover:bg-red-500 transition-all hover:text-white"
          onClick={recognizeSpeech}
        >
          <FaMicrophone/>
        </button>
        <button 
          className="rounded-xl px-4 bg-blue-400 shadow-md cursor-pointer hover:bg-blue-500 transition-all hover:text-white"
          onClick={recognizeBluetooth}
        >
          <FaBluetoothB/>
        </button>
        <button 
          className={`rounded-xl px-4 ${!voice ? "bg-green-400" : "bg-green-500 text-white"} shadow-md cursor-pointer hover:bg-green-500 transition-all hover:text-white`}
          onClick={() => setVoice(p => !p)}
        >
          <MdRecordVoiceOver />
        </button>
      </div>
      <div className="px-4 py-2 rounded-xl w-1/2 min-h-10 text-gray-500 mb-4 font-mono tracking-tight min-w-128">
        {input}
      </div>
      <div className="w-1/2 min-h-1/2 rounded-xl p-4 shadow-xl min-w-128 font-mono tracking-tight">
        <Markdown>{output}</Markdown>
      </div>
    </div>
  );
}

export default App;
