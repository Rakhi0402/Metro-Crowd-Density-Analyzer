"""
dashboard.py
-----------
Generates interactive dashboard and HTML reports for crowd forecasting.
Provides visual alerts and operator recommendations.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json


class DashboardReport:
    """Generates HTML dashboard for crowd monitoring and predictions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html(self, coaches_data: list[dict], alerts_data: list[dict], 
                      timestamp: str) -> Path:
        """
        Generate unified interactive HTML dashboard with coach selection.
        
        Parameters
        ----------
        coaches_data : list[dict]
            List of coach analysis data.
        alerts_data : list[dict]
            List of alert entries.
        timestamp : str
            Session timestamp.
        
        Returns
        -------
        Path
            Path to generated HTML file.
        """
        # Use static filename so it updates in place
        html_path = self.output_dir / "dashboard.html"
        
        # Create coach selector options
        coach_options = ""
        coaches_json = {}
        risk_summary = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        
        for coach in coaches_data:
            coach_id = coach['coach_id']
            coach_options += f'<option value="coach-{coach_id}">Coach {coach_id}</option>\n'
            coaches_json[f"coach-{coach_id}"] = coach
            
            risk = coach.get('risk_level', 'LOW')
            risk_summary[risk] = risk_summary.get(risk, 0) + 1
        
        # Build summary cards
        summary_html = ""
        for risk, count in risk_summary.items():
            risk_class = risk.lower()
            summary_html += f"""
            <div class="summary-card risk-{risk_class}">
                <strong>{count}</strong>
                <span>{risk.title()} risk coaches</span>
                <div class="summary-pill">{risk}</div>
            </div>
            """
        
        # Build alerts section
        alert_items = ""
        for alert in alerts_data:
            severity_icons = {
                'critical': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️'
            }
            icon = severity_icons.get(alert.get('level', 'info'), '•')
            ts = alert.get('timestamp') or alert.get('time', 'N/A')
            msg = alert.get('message') or alert.get('msg', 'No message')
            level = alert.get('level', 'info')
            
            alert_items += f"""
            <div class="alert alert-{level}">
                <span class="icon">{icon}</span>
                <span class="timestamp">{ts}</span>
                <span class="message">{msg}</span>
            </div>
            """
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metro Crowd Monitoring Dashboard</title>
    <style>
        :root {
            --bg: #f7f9ff;
            --surface: #ffffff;
            --surface-soft: #f1f5ff;
            --text: #15213a;
            --muted: #5f6d85;
            --accent: #4f74ff;
            --accent-deep: #2b5bff;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --info: #17a2b8;
            --radius: 20px;
            --shadow: 0 20px 50px rgba(31, 45, 110, 0.08);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            background: radial-gradient(circle at top left, rgba(79,116,255,0.18), transparent 30%),
                        radial-gradient(circle at bottom right, rgba(255,192,69,0.16), transparent 34%),
                        linear-gradient(180deg, #eef5ff 0%, #f7fbff 100%);
            color: var(--text);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 24px;
        }

        .container {
            max-width: 1320px;
            margin: 0 auto;
            display: grid;
            gap: 22px;
        }

        header {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 28px 32px;
            box-shadow: var(--shadow);
            display: grid;
            gap: 18px;
        }

        header h1 {
            font-size: clamp(2rem, 2.8vw, 2.6rem);
            letter-spacing: -0.03em;
        }

        header .timestamp {
            color: var(--muted);
            font-size: 0.96rem;
        }

        .controls-panel,
        .summary-section,
        .content-panel,
        .alerts-section {
            background: var(--surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 24px;
        }

        .controls-panel {
            display: grid;
            gap: 18px;
        }

        .coach-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            align-items: center;
        }

        .coach-selector label {
            font-weight: 700;
            color: var(--text);
        }

        .coach-selector select {
            min-width: 260px;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #dfe5ff;
            background: #ffffff;
            color: var(--text);
            font-size: 0.98rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .coach-selector select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 4px rgba(79,116,255,0.12);
        }

        .button-group {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .button-group button {
            border: none;
            border-radius: 16px;
            padding: 13px 20px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .button-group button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(79,116,255,0.15);
        }

        .button-primary {
            background: linear-gradient(135deg, var(--accent), var(--accent-deep));
            color: #fff;
        }

        .button-secondary {
            background: #f6f8ff;
            color: var(--text);
        }

        .summary-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
        }

        .summary-card {
            padding: 22px;
            border-radius: 18px;
            background: var(--surface-soft);
            border: 1px solid #eef3ff;
            min-height: 130px;
            display: grid;
            gap: 10px;
        }

        .summary-card strong {
            font-size: 1.6rem;
            color: var(--text);
        }

        .summary-card span {
            color: var(--muted);
        }

        .summary-card .summary-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 14px;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .summary-card.risk-critical .summary-pill { background: rgba(220,53,69,0.12); color: var(--danger); }
        .summary-card.risk-high .summary-pill { background: rgba(255,193,7,0.14); color: var(--warning); }
        .summary-card.risk-moderate .summary-pill { background: rgba(23,162,184,0.14); color: var(--info); }
        .summary-card.risk-low .summary-pill { background: rgba(40,167,69,0.14); color: var(--success); }

        .content-panel {
            display: grid;
            gap: 24px;
        }

        #coaches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 18px;
        }

        .coach-card {
            padding: 22px;
            border-radius: 20px;
            background: var(--surface);
            border: 1px solid #eef3ff;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            cursor: pointer;
        }

        .coach-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 28px rgba(79,116,255,0.12);
        }

        .coach-card h3 {
            margin-bottom: 16px;
        }

        .coach-line {
            display: grid;
            gap: 6px;
            margin-bottom: 16px;
        }

        .coach-line span {
            color: var(--muted);
            font-size: 0.85rem;
        }

        .coach-line strong {
            color: var(--text);
            font-size: 1.1rem;
        }

        .coach-detail {
            display: none;
            padding: 24px;
        }

        .coach-detail.active {
            display: block;
        }

        .coach-detail h2 {
            margin-bottom: 24px;
            font-size: 1.4rem;
        }

        .metric {
            padding: 18px 20px;
            border-radius: 18px;
            background: #f5f7ff;
            border-left: 4px solid var(--accent);
        }

        .recommendation {
            padding: 22px;
            background: rgba(45, 111, 255, 0.08);
            border-left: 5px solid var(--info);
        }

        .forecast-section {
            padding: 20px 22px;
            background: rgba(255, 193, 7, 0.1);
            border-left: 5px solid var(--warning);
        }

        .forecast-values {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
            gap: 12px;
        }

        .forecast-item {
            border-radius: 14px;
            padding: 12px 14px;
            border: 1px solid #ffe7b7;
            text-align: center;
            font-weight: 700;
            color: var(--text);
            background: var(--surface);
        }

        .alerts-section {
            display: grid;
            gap: 18px;
        }

        .alerts-section h2 {
            font-size: 1.35rem;
        }

        .alert {
            grid-template-columns: auto 1fr;
            gap: 14px;
            align-items: center;
            border-radius: 18px;
            border: 1px solid #eef3ff;
            background: var(--surface-soft);
        }

        .alert .icon {
            width: 44px;
            height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: #eef5ff;
        }

        footer {
            text-align: center;
            color: var(--muted);
            padding: 16px 0;
            font-size: 0.95rem;
        }

        @media (max-width: 860px) {
            .coach-selector,
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🚆 Metro Train Crowd Monitoring Dashboard</h1>
                <p class="timestamp">Updated: __TIMESTAMP__</p>
            </div>
            <p class="timestamp">Review coach crowding and risk levels across the latest batch. Click a coach card or use the filters to focus on the most important data.</p>
        </header>

        <section class="controls-panel">
            <div class="coach-selector">
                <label for="coach-select">Jump to coach</label>
                <select id="coach-select" onchange="switchCoach()">
                    <option value="">-- Select a Coach --</option>
                    __COACH_OPTIONS__
                </select>
            </div>
            <div class="button-group">
                <button class="button-primary" onclick="showAllCoaches()">Show All Coaches</button>
                <button class="button-secondary" onclick="filterRisk('CRITICAL')">Critical</button>
                <button class="button-secondary" onclick="filterRisk('HIGH')">High</button>
                <button class="button-secondary" onclick="filterRisk('MODERATE')">Moderate</button>
                <button class="button-secondary" onclick="filterRisk('LOW')">Low</button>
            </div>
        </section>

        <section class="summary-section">
            <div class="summary-card">
                <strong>__TOTAL_COACHES__</strong>
                <span>Total Coaches Processed</span>
            </div>
            <div class="summary-card">
                <strong>__TOTAL_ALERTS__</strong>
                <span>Alerts Generated</span>
            </div>
            __SUMMARY_HTML__
        </section>

        <section class="content-panel">
            <div id="all-coaches" class="coach-detail active">
                <h2>📊 All Coaches Overview</h2>
                <div id="coaches-grid"></div>
            </div>
        """

        html_content = html_content.replace("__TIMESTAMP__", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        html_content = html_content.replace("__COACH_OPTIONS__", coach_options)
        html_content = html_content.replace("__TOTAL_COACHES__", str(len(coaches_data)))
        html_content = html_content.replace("__TOTAL_ALERTS__", str(len(alerts_data)))
        html_content = html_content.replace("__SUMMARY_HTML__", summary_html)

        # Add individual coach detail sections
        for coach in coaches_data:
            coach_id = coach['coach_id']
            risk = coach.get('risk_level', 'LOW')
            forecast_str = ", ".join([f"{p:.0f}" for p in coach['forecast'][:5]])
            
            # Determine color based on risk level
            color_map = {
                'CRITICAL': '#dc3545',
                'HIGH': '#fd7e14',
                'MODERATE': '#ffc107',
                'LOW': '#28a745'
            }
            risk_color = color_map.get(risk, '#666')
            
            html_content += f"""
        <div id="coach-{coach_id}" class="coach-detail">
            <h2>Coach {coach_id} - {coach.get('video', 'Unknown')}</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="label">Current Average</span>
                    <span class="value">{coach.get('current_avg', 0):.1f} people</span>
                </div>
                <div class="metric">
                    <span class="label">Current Peak</span>
                    <span class="value">{coach.get('current_max', 0)} people</span>
                </div>
                <div class="metric">
                    <span class="label">Risk Level</span>
                    <span class="value risk">{risk}</span>
                </div>
                <div class="metric">
                    <span class="label">Status</span>
                    <span class="value" style="font-size: 16px; color: {risk_color};">{risk}</span>
                </div>
            </div>
            
            <div class="recommendation">
                <strong>Operator Recommendation:</strong>
                <p>{coach.get('recommendation', 'Monitor situation')}</p>
            </div>
            
            <div class="forecast-section">
                <strong>📈 5-Minute Forecast:</strong>
                <div class="forecast-values">
                    {' '.join(f'<div class="forecast-item">{int(p)}</div>' for p in coach['forecast'][:5])}
                </div>
            </div>
        </div>
"""
        
        html_content += f"""
        <div class="alerts-section">
            <h2>🚨 Recent Alerts ({len(alerts_data)})</h2>
            {alert_items if alert_items else '<p style="color: #999;">No alerts generated</p>'}
        </div>
        
        <footer>
            <p>Metro Crowd Density Tracking System | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    
    <script>
        const coachesData = {{
"""
        
        for coach in coaches_data:
            coach_id = coach['coach_id']
            html_content += f'            "coach-{coach_id}": {json.dumps(coach, default=str)},\n'
        
        html_content += """        };
        
        function switchCoach() {
            const selector = document.getElementById('coach-select');
            const coachId = selector.value;

            document.querySelectorAll('.coach-detail').forEach(el => el.classList.remove('active'));

            if (!coachId) {
                showAllCoaches();
                return;
            }

            const element = document.getElementById(coachId);
            if (element) {
                element.classList.add('active');
                element.scrollIntoView({ behavior: 'smooth' });
            }
        }
        
        function showAllCoaches() {
            document.querySelectorAll('.coach-detail').forEach(el => el.classList.remove('active'));
            document.getElementById('all-coaches').classList.add('active');
            document.getElementById('coach-select').value = '';

            const grid = document.getElementById('coaches-grid');
            grid.innerHTML = '';

            Object.entries(coachesData).forEach(([coachKey, coach]) => {
                const risk = coach.risk_level || 'LOW';
                const forecastValues = Array.isArray(coach.forecast) ? coach.forecast : [];
                const forecast = forecastValues.slice(0, 5).map(p => Math.round(p)).join(', ');
                const riskColor = risk.includes('CRITICAL') ? '#dc3545'
                    : risk.includes('HIGH') ? '#fd7e14'
                    : risk.includes('MODERATE') ? '#17a2b8'
                    : '#28a745';
                const avg = Number(coach.current_avg || 0);
                const peak = Number(coach.current_max || 0);

                const card = document.createElement('div');
                card.className = 'coach-card';
                card.style.borderLeft = '4px solid ' + riskColor;
                card.onclick = () => {
                    document.getElementById('coach-select').value = coachKey;
                    switchCoach();
                };

                card.innerHTML =
                    '<h3>Coach ' + coach.coach_id + '</h3>' +
                    '<div class="coach-line"><span>Average</span><strong>' + avg.toFixed(1) + ' people</strong></div>' +
                    '<div class="coach-line"><span>Peak</span><strong>' + peak + ' people</strong></div>' +
                    '<div class="coach-line"><span>Risk</span><strong style="color:' + riskColor + ';">' + risk + '</strong></div>' +
                    '<div class="coach-line"><span>Forecast</span><strong>' + forecast + '</strong></div>';

                grid.appendChild(card);
            });
        }

        function filterRisk(level) {
            document.querySelectorAll('.coach-detail').forEach(el => el.classList.remove('active'));
            document.getElementById('all-coaches').classList.add('active');
            document.getElementById('coach-select').value = '';

            const grid = document.getElementById('coaches-grid');
            grid.innerHTML = '';

            Object.entries(coachesData).forEach(([coachKey, coach]) => {
                if ((coach.risk_level || 'LOW') === level) {
                    const risk = coach.risk_level || 'LOW';
                    const forecastValues = Array.isArray(coach.forecast) ? coach.forecast : [];
                    const forecast = forecastValues.slice(0, 5).map(p => Math.round(p)).join(', ');
                    const riskColor = risk.includes('CRITICAL') ? '#dc3545'
                        : risk.includes('HIGH') ? '#fd7e14'
                        : risk.includes('MODERATE') ? '#17a2b8'
                        : '#28a745';
                    const avg = Number(coach.current_avg || 0);
                    const peak = Number(coach.current_max || 0);

                    const card = document.createElement('div');
                    card.className = 'coach-card';
                    card.style.borderLeft = '4px solid ' + riskColor;
                    card.onclick = () => {
                        document.getElementById('coach-select').value = coachKey;
                        switchCoach();
                    };

                    card.innerHTML =
                        '<h3>Coach ' + coach.coach_id + '</h3>' +
                        '<div class="coach-line"><span>Average</span><strong>' + avg.toFixed(1) + ' people</strong></div>' +
                        '<div class="coach-line"><span>Peak</span><strong>' + peak + ' people</strong></div>' +
                        '<div class="coach-line"><span>Risk</span><strong style="color:' + riskColor + ';">' + risk + '</strong></div>' +
                        '<div class="coach-line"><span>Forecast</span><strong>' + forecast + '</strong></div>';
                    grid.appendChild(card);
                }
            });
        }
        
        // Initialize with all coaches view
        showAllCoaches();
    </script>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def generate_json_report(self, coaches_data: list[dict], alerts_data: list[dict],
                            timestamp: str) -> Path:
        """
        Generate JSON report for integration with external systems.
        
        Parameters
        ----------
        coaches_data : list[dict]
            List of coach analysis data.
        alerts_data : list[dict]
            List of alert entries.
        timestamp : str
            Session timestamp.
        
        Returns
        -------
        Path
            Path to generated JSON file.
        """
        json_path = self.output_dir / "report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_coaches': len(coaches_data),
            'total_alerts': len(alerts_data),
            'coaches': coaches_data,
            'alerts': alerts_data,
            'risk_summary': {
                'critical': sum(1 for c in coaches_data if 'CRITICAL' in c.get('risk_level', '')),
                'high': sum(1 for c in coaches_data if 'HIGH' in c.get('risk_level', '')),
                'moderate': sum(1 for c in coaches_data if 'MODERATE' in c.get('risk_level', '')),
                'low': sum(1 for c in coaches_data if c.get('risk_level', '') == 'LOW')
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        return json_path
