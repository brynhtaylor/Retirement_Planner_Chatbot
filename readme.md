# AI‑Assisted Retirement Planner

**An interactive, AI-powered retirement planning tool built using Dash, Plotly, and Ollama.**  
It allows users to dynamically simulate their financial future while asking an integrated AI assistant for tailored retirement advice based on their specific plan.

---

## 📋 Features

- **Dynamic Glidepath Projections:**  
  Choose between a standard glidepath or a historically optimized, data-driven glidepath based on real-world asset returns.

- **Monte Carlo Simulations:**  
  Robust modeling of portfolio growth including personalized monthly withdrawals in retirement.

- **AI Retirement Assistant:**  
  Embedded chatbot connected to your personal simulation results, allowing for personalized answers and planning guidance.

- **Modern, Responsive UI:**  
  Dark mode, Open Sans/Inter fonts, professional visual theming, and mobile-friendly layout.

---

## 📦 Requirements

- **Python:** 3.12.6
- **Ollama:** Local LLM API server ([Install Guide](https://ollama.com/))
  - Model Required: `gemma3`
- **Python Packages:** Listed in `requirements.txt`
  - (Key packages: `dash`, `plotly`, `scipy`, `numpy`, `pandas`, `ollama`)

---

## 📁 Project Structure

```bash
/
├── data/
│   └── processed/
│       └── cleaned_returns.csv  # Generated after running get_data.py
├── prompts/
│   └── chatbot_context.md       # System prompt for the AI assistant
├── code/
│   ├── get_data.py               # Script to download & preprocess return data
│   ├── chatbot.py                # Harness for interacting with Ollama
│   └── planner.py                # Main Dash application
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. **Install prerequisites**

   Make sure you have Python 3.12.6 and Ollama installed.

2. **Set up Python environment**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess financial data**

   Run:

   ```bash
   python code/get_data.py
   ```

   This will create the file `data/processed/cleaned_returns.csv`, which is required for the simulation engine.

4. **Start Ollama server**

   Open a terminal and run:

   ```bash
   ollama run gemma3
   ```

   (Ensure the model is downloaded and ready.)

5. **Launch the Retirement Planner application**

   ```bash
   python code/planner.py
   ```

   The app will be available at `http://127.0.0.1:8050/`.

---

## 🛠️ Customization Notes

- **Glidepath Engine:**  
  You can modify `generate_historical_optimized_glidepath()` in `planner.py` if you wish to customize the optimization logic further.

- **AI Assistant Prompt:**  
  To edit the behavior and tone of the chatbot, update `prompts/chatbot_context.md`.

- **Model Replacement:**  
  You can swap `gemma3` in `chatbot.py` for another model available in Ollama, depending on your machine's capabilities.

---

## ⚡ Key Technical Highlights

- **Session-Aware Chatbot:**  
  Chatbot dynamically pulls in the user's most recent simulation to craft context-specific responses.

- **Two-Asset Portfolio Simulation:**  
  Asset returns are drawn from multivariate normal distributions calibrated on cleaned historical data.

- **Seamless UI:**  
  Modern two-column layout with flexible resizing, dark theme, and smart interaction patterns (including animated chatbot behavior).

---

## 📜 License

This project is released for **educational and non-commercial use only**.  
Feel free to adapt and extend it for personal or learning purposes!