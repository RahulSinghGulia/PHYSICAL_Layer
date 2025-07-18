# MD-OFDM: Energy-Efficient & Low-PAPR MIMO-OFDM for Resource-Constrained Applications

[![License: CC BY](https://img.shields.io/badge/License-CC_BY-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![GitHub last commit](https://img.shields.io/github/last-commit/YourGitHubUsername/your-repo-name)](https://github.com/YourGitHubUsername/your-repo-name/commits/main)
---

## 🚀 Project Overview

**MD-OFDM (Multi-Dimensional Orthogonal Frequency Division Multiplexing)** is a novel variant of the widely used MIMO-OFDM system, specifically engineered to address two critical challenges in modern wireless communications: **high Peak-to-Average Power Ratio (PAPR)** and **significant power consumption** due to multiple active Radio Frequency (RF) chains.

Traditional MIMO-OFDM, while offering high data rates through spatial multiplexing, often requires complex hardware and consumes considerable power. MD-OFDM proposes a smarter approach: **per-subcarrier transmit antenna selection**. Instead of activating all transmit antennas for every subcarrier, MD-OFDM intelligently selects only the *best* single transmit antenna for each individual subcarrier.

This project provides a detailed mathematical model, analysis, and simulation results demonstrating how MD-OFDM achieves:

* **Significantly lower PAPR**, simplifying power amplifier design.
* **Reduced absolute power consumption**, ideal for battery-powered devices.
* **Improved Bit Error Rate (BER)** performance, enhancing link reliability.

While traditional MIMO-OFDM excels in maximizing spectral efficiency, MD-OFDM offers a compelling trade-off, prioritizing **energy conservation, hardware simplicity, and robust communication** for a new generation of wireless applications.

---

## ✨ Key Features & Advantages

* **Per-Subcarrier Transmit Antenna Selection:** Dynamically selects the optimal transmit antenna for each subcarrier based on channel conditions, minimizing interference and maximizing gain.
* **Reduced PAPR:** By limiting simultaneous transmissions from multiple antennas per subcarrier, MD-OFDM effectively mitigates high power peaks, allowing for simpler, more efficient, and less costly power amplifiers.
* **Enhanced Energy Efficiency:** Minimizes the number of active RF chains, leading to a substantial reduction in overall power consumption, crucial for power-constrained devices and "Green Communications."
* **Superior BER Performance:** Simulations demonstrate MD-OFDM's improved robustness against noise, resulting in a lower Bit Error Rate compared to conventional MMSE MIMO.
* **Suitability for Resource-Constrained Applications:** Designed with the needs of IoT, LPWAN, and mMTC in mind, where hardware cost, battery life, and link reliability are paramount.

---

## ⚙️ How It Works (Simplified)

At its core, MD-OFDM operates by making intelligent decisions at the transmitter:

1.  **Smart Selection:** For every individual frequency subcarrier, the transmitter evaluates the channel conditions to each available transmit antenna.
2.  **Single Active Antenna:** Based on this evaluation (e.g., selecting the antenna with the strongest channel gain), only *one* transmit antenna is activated to send data for that specific subcarrier. All other antennas remain idle for that subcarrier.
3.  **Simplified Reception:** At the receiver, the signal from the selected transmit antenna is processed, often with simpler equalization compared to multi-stream MIMO.

This focused approach contrasts with traditional MIMO-OFDM, which simultaneously uses all transmit antennas across all subcarriers.

---

## 📊 Simulation Results & Insights

The project includes detailed mathematical models for BER, Energy Efficiency (EE), and PAPR. Simulations confirm the theoretical advantages:

* **BER:** MD-OFDM consistently shows a **lower BER** across various Signal-to-Noise Ratio (SNR) levels, indicating a more reliable communication link.
* **PAPR:** MD-OFDM exhibits a **significantly lower PAPR** (demonstrated through CCDF curves), which directly translates to less stringent requirements for power amplifier linearity and lower hardware costs.
* **Energy Efficiency:** While MMSE MIMO can achieve higher *peak spectral efficiency*, MD-OFDM achieves **lower absolute power consumption**. The overall EE trade-off highlights MD-OFDM's strength in scenarios where power saving is prioritized over raw peak data rate.

---

## 🎯 Target Applications

MD-OFDM is particularly well-suited for deployments where hardware simplicity, operational longevity, and robust, lower-rate communication are key drivers:

* **Internet of Things (IoT) & Wireless Sensor Networks (WSN):** Extend battery life and simplify device hardware.
* **Low-Power Wide Area Networks (LPWAN):** Align with goals of long battery life and wide coverage for massive deployments.
* **Massive Machine-Type Communications (mMTC):** Efficient uplink or control channels for a huge number of connected devices.
* **Device-to-Device (D2D) Communications:** Enable direct, energy-efficient communication between devices.
* **Cost-Sensitive Wireless Systems:** Reduce manufacturing costs by relaxing power amplifier linearity requirements.

---

## 🚀 Getting Started

To explore the MD-OFDM model and replicate the simulation results:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/your-repo-name.git](https://github.com/YourGitHubUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    * **[If Python/MATLAB code, specify requirements here, e.g.:]**
        * For Python simulations: Ensure you have Python 3.x installed. Install required libraries:
            ```bash
            pip install numpy scipy matplotlib
            ```
        * For MATLAB simulations: MATLAB R20XXa or newer. No specific toolbox installations beyond standard ones are usually required for basic signal processing.

3.  **Run the simulations:**
    * **[Specify how to run your main simulation scripts, e.g.:]**
        * For Python:
            ```bash
            python simulation_main.py
            ```
        * For MATLAB:
            Open MATLAB and navigate to the `your-repo-name/code` directory. Then run:
            ```matlab
            main_simulation_script
            ```
    * Simulation results (plots, data) will be generated in `results/` or similar.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, find a bug, or want to add new features, please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. You are free to share and adapt the material for any purpose, even commercially, as long as you give appropriate credit to the original author(s).

---

## 📞 Contact

For any questions or collaborations, feel free to reach out:

**Rahul Gulia** Rochester Institute of Technology  
Email: rg9828@rit.edu

---
