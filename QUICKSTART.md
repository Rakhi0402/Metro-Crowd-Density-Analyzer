# Quick Start Guide - Crowd Prediction & Alerts

## 🚀 Get Started in 2 Minutes

### 1️⃣ Run Batch Processing with Predictions

```powershell
cd "d:\metro crowding\Crowd-density-tracking-main"
& ".\.venv\Scripts\python.exe" batch_processor.py --headless
```

**This will:**

- ✅ Process all 20 videos automatically
- ✅ Predict future crowd levels
- ✅ Detect overcrowding situations
- ✅ Generate alerts
- ✅ Create reports and dashboard
- ✅ **Automatically learn from new data** (retrain AI model)
- ✅ **Remove redundant data** (clean old logs)

---

## 📊 View Your Results

After the batch processor completes, open:

📁 **outputs/** folder → Look for:

1. **`dashboard.html`** ← **OPEN THIS IN YOUR BROWSER** 🌐
   - **Single unified dashboard** with all coaches
   - Select which coach to view with dropdown menu
   - View all coaches overview
   - Interactive risk levels
   - Real-time alerts
   - Operator recommendations
   - **Updates with each run** (same file)

2. **`batch_summary_*.txt`**
   - Detailed text report
   - All statistics and forecasts

3. **`batch_report_*.csv`**
   - Raw data in spreadsheet format

4. **`report_*.json`**
   - Machine-readable format

---

## 🚨 Understanding Alerts

### Alert Levels (from Console Output)

```
🚨 CRITICAL   → Peak crowding > 20 people (action needed now!)
🟠 HIGH       → Peak crowding > 15 people (monitor closely)
🟡 MODERATE   → Peak crowding > 10 people (normal caution)
🟢 LOW        → Normal passenger flow (standard ops)
```

### Example Alert Messages

```
⚠️  [13:55:45] HIGH CROWDING DETECTED: 16 people
📈 [13:56:12] INCREASING TREND: Expect up to 18 people
🚨 [13:57:20] OVERCROWDING FORECAST: Peak predicted at 21 people
```

---

## 🎯 Common Commands

| Command                                     | What It Does                              |
| ------------------------------------------- | ----------------------------------------- |
| `batch_processor.py --headless`             | Process all videos, generate full reports |
| `batch_processor.py --headless --limit 5`   | Test with first 5 videos only             |
| `batch_processor.py --headless --loop`      | Run continuously (repeats forever)        |
| `batch_processor.py --headless --algo flow` | Use optical flow algorithm instead        |

---

## 🧠 Automatic Learning & Data Management

Your system gets smarter with every run!

### How It Learns

- **After each batch processing**, the system automatically:
  - Retrains the AI forecasting model using new crowd data
  - Improves predictions for future crowd levels
  - Adapts to changing patterns (rush hours, events, etc.)

- **Data Cleanup**: Removes old/redundant data to keep the system efficient
  - Keeps data from the last 30 days
  - Removes duplicate entries
  - Maintains optimal performance

### Benefits

- 📈 **Better Predictions**: More accurate crowd forecasts over time
- 🔄 **Self-Improving**: Learns from real-world usage
- 🧹 **Clean Data**: No manual maintenance needed
- ⚡ **Always Current**: Model stays updated with latest patterns

---

### Console Output Example

```
[Batch] Initializing YOLO processor...
[PersonDetector] Model ready.
[Batch] Found 20 video(s) to process.

======================================================================
ITERATION 1
======================================================================

[Coach 1] Processing video1.mp4...
  Frame 240: 11 people | 2.7 FPS

  ✓ Coach 1 complete: 350 frames, avg 11.7 people, 5.6 FPS

[Coach 2] Processing video10.mp4...
  ...

======================================================================
CROWD PREDICTIONS & ALERTS:
======================================================================

Coach 1: video1.mp4
  Current Average:  11.7 people
  Current Peak:     15 people
  Risk Level:       🟡 MODERATE
  Next 5 Frames:    ['12', '13', '12', '11', '10']
  Recommendation:   🟡 MODERATE: Monitor closely, have contingency ready

Coach 2: video10.mp4
  ...
```

---

## 🌐 Dashboard Features

Open the **HTML dashboard** to see:

### Risk Summary

- How many coaches at each danger level
- Visual traffic light indicators

### Coach Cards

- Current crowd levels
- Peak values
- Predicted next 5 minutes
- Specific actions to take

### Alert Feed

- Real-time alert history
- Color-coded by severity
- Timestamps for each event

---

## ⚙️ Enable Email/SMS Alerts (Optional)

### For Email:

1. Edit `config/alerts_config.json`
2. Add Gmail credentials
3. Run with:
   ```powershell
   batch_processor.py --headless --alerts email
   ```

### For SMS (Twilio):

1. Edit `config/alerts_config.json`
2. Add Twilio credentials
3. Run with:
   ```powershell
   batch_processor.py --headless --alerts sms
   ```

---

## 📋 Output Files Explained

After running, check the **`outputs/`** folder:

| File                  | Purpose                                        |
| --------------------- | ---------------------------------------------- |
| `dashboard_*.html`    | 🌐 Interactive web dashboard (open in browser) |
| `batch_summary_*.txt` | 📄 Human-readable report with all details      |
| `batch_report_*.csv`  | 📊 Raw data for spreadsheets                   |
| `alerts_*.csv`        | 🚨 Log of all alerts generated                 |
| `report_*.json`       | 🔗 For integration with other systems          |

---

## 🔍 How Predictions Work

**Your system forecasts:**

- Next 5 minutes of crowd levels
- Whether crowding is increasing/decreasing
- Risk level changes
- When overcrowding will occur

**Used for:**

- Alerting operators before problems happen
- Pre-positioning staff
- Rerouting passengers
- Preventing dangerous overcrowding

---

## 💡 Tips

✅ **Best Practice**: Run batch processor regularly (daily/weekly)  
✅ **Automate**: Use Windows Task Scheduler for scheduled runs  
✅ **Monitor**: Check outputs folder periodically  
✅ **Adjust**: Lower alert threshold if you want more early warnings  
✅ **Test**: Try `--limit 2` first to see how it works

---

## 🆘 Troubleshooting

| Problem                | Solution                                                 |
| ---------------------- | -------------------------------------------------------- |
| No alerts?             | They're saved in `alerts_*.csv`. Check outputs folder.   |
| Dashboard not opening? | Use Firefox or Chrome. Right-click HTML file → Open With |
| Processing too slow?   | Add `--algo flow` to try optical flow (faster)           |
| Want fewer alerts?     | Increase threshold in forecaster (contact developer)     |

---

## 📞 System Components

```
Your Video Files
       ↓
[YOLO Detection] → Count people in each frame
       ↓
[Forecasting] → Predict next 5 minutes
       ↓
[Risk Assessment] → Determine danger level
       ↓
[Alerts] → Notify if overcrowding detected
       ↓
[Dashboard] → Show everything on web page
```

---

## 🎬 Next Steps

1. **Run it now:**

   ```powershell
   batch_processor.py --headless --limit 2
   ```

2. **Open the dashboard:**
   - Find `outputs/dashboard_*.html`
   - Double-click to open in browser

3. **Check the alerts:**
   - See what crowding was detected
   - View predictions
   - Get operator recommendations

4. **Automate it:**
   - Schedule with Windows Task Scheduler
   - Run every hour/day as needed

---

## ✨ Features Summary

✅ Real-time people detection  
✅ Crowd level prediction (5 min ahead)  
✅ Automatic overcrowding alerts  
✅ Multi-coach train simulation  
✅ Interactive HTML dashboard  
✅ Email/SMS/Slack notifications  
✅ CSV, JSON, Text reports  
✅ Continuous monitoring mode

---

**You're ready!** Run the batch processor and check the dashboard. 🎉
