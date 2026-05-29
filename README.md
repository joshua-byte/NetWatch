# NetWatch - A Network Anomaly Detection System

> A Python-based network anomaly detection system using adaptive statistical baselining, live packet capture, and event extraction inspired by measure-theoretic signal decomposition.

---

## Dashboard Preview

<img width="1905" height="896" alt="dashboard-overview" src="https://github.com/user-attachments/assets/0829dc77-446e-4eb3-8bb8-50d0a1c5fb33" />


---

## Features

### Stable Features

* CSV-based anomaly detection
* Synthetic traffic demo mode
* Adaptive rolling baseline (Median + MAD)
* Statistical anomaly scoring
* Event extraction and severity classification
* Interactive Streamlit dashboard
* Plotly visualizations
* PDF report generation
* Exportable event tables

### Experimental Features

* Live packet capture using Scapy
* Real-time network traffic analysis
* Live anomaly alerting
* Traffic profiling and event correlation

> **Note:** The live capture pipeline is currently under active development. CSV and synthetic analysis modes are the primary supported workflows.

---

## Project Structure

```text
anomaly-detector/
│
├── app.py
├── detector.py
├── capture.py
├── dashboard.py
├── report.py
│
├── screenshots/
│   ├── dashboard-overview.png
│   ├── events-table.png
│   ├── statistics-view.png
│   └── report-preview.png
│
├── sample_data/
│
├── requirements.txt
└── README.md
```

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

---

## Input Modes

### CSV Analysis

Analyze traffic datasets such as CICIDS2017 or any CSV containing numerical network features.

Example:

```text
Friday-WorkingHours-Morning.pcap_ISCX.csv
```

### Synthetic Demo

Built-in traffic generator with injected anomalies for testing and demonstrations.

### Live Capture (Experimental)

Real-time packet capture using Scapy.

```bash
sudo python capture.py --iface eth0
```

---

## Detection Methodology

The detector models network activity as a combination of a continuous background process and sparse anomalous events:

```text
x(t) = f(t)dt + Σ aᵢ δ(t − tᵢ)
```

Where:

* **f(t)** represents normal traffic behaviour
* **σ(t)** is estimated using robust statistics
* Significant deviations generate support regions
* Support regions collapse into sparse anomaly events

This approach focuses on extracting meaningful events from noisy traffic streams rather than treating all fluctuations as anomalies.

---

## Detection Pipeline

| Traversal | Component         | Purpose                                       |
| --------- | ----------------- | --------------------------------------------- |
| 1         | Robust Background | Rolling median and MAD estimation             |
| 2         | Deviation Field   | Z-score based anomaly scoring                 |
| 3         | Event Extraction  | Collapse anomaly regions into discrete events |

---

## Severity Levels

| Severity    | Threshold |
| ----------- | --------- |
| 🟢 Low      | ≥ 3σ      |
| 🟡 Medium   | ≥ 5σ      |
| 🟠 High     | ≥ 7σ      |
| 🔴 Critical | ≥ 10σ     |

---

## Outputs

The system provides:

* Interactive anomaly visualizations
* Deviation field analysis
* Singular measure decomposition plots
* Severity distribution charts
* Filterable anomaly event tables
* CSV exports
* PDF reports

---

## Screenshots

### Live Dashboard

<img width="1905" height="896" alt="dashboard-overview" src="https://github.com/user-attachments/assets/524aff55-f78f-4612-b656-2d40ed3bdec4" />


### Event Analysis

<img width="1905" height="896" alt="events" src="https://github.com/user-attachments/assets/eaabb1a2-9712-45cc-b87f-ada918dfbe0d" />


### Statistical Analysis

<img width="1905" height="896" alt="stats" src="https://github.com/user-attachments/assets/65684dd1-a96e-4443-8cea-41f7325100d1" />


### PDF Reporting

<img width="796" height="826" alt="report" src="https://github.com/user-attachments/assets/35be6b83-2990-45ed-aa43-692cf38f45e6" />


---

## Example Results

| Scenario                  | Expected Outcome         |
| ------------------------- | ------------------------ |
| Normal Traffic            | No significant anomalies |
| Traffic Burst             | Medium severity event    |
| Sustained Spike           | High severity event      |
| Injected Synthetic Attack | Critical event           |

---

## Technologies Used

* Python
* Streamlit
* Scapy
* Plotly
* Pandas
* NumPy
* ReportLab

---

## Current Status

| Module              | Status          |
| ------------------- | --------------- |
| CSV Analysis        | ✅ Stable        |
| Synthetic Demo      | ✅ Stable        |
| Dashboard           | ✅ Stable        |
| Report Generation   | ✅ Stable        |
| Live Capture        | 🚧 Experimental |
| Real-Time Detection | 🚧 Experimental |

---

## Future Work

* Multi-feature anomaly detection
* Flow-based traffic analysis
* Real-time alerting pipeline
* DDoS classification
* Historical trend analysis
* Machine learning-assisted anomaly scoring

---

## Author

**Joshua**
RV University, Bengaluru

Cybersecurity • Systems Analysis • Philosophy of Science

---

## License

MIT License
